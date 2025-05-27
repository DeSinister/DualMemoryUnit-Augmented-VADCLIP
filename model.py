from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from cross_former import Crossformer
import math
from torch.nn.modules.module import Module

class Memory_Unit(Module):
    """
    Memory unit module for storing and attending to memory blocks.
    """
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim
        self.nums = nums
        # Learnable memory blocks of shape (nums, dim)
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize memory block parameters uniformly.
        """
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)
    
       
    def forward(self, data):  ####data size---> B,T,D       K,V size--->K,D
        """
        Forward pass.
        Args:
            data: Tensor of shape (B, T, D)
        Returns:
            temporal_att: Attention scores aggregated (B, T)
            augment: Memory-augmented features (B, T, D)
        """
        # Compute attention scores scaled by sqrt(dim)
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))   #### Att---> B,T,K
        # Aggregate top attention scores over the last dimension
        temporal_att = torch.topk(attention, self.nums//16+1, dim = -1)[0].mean(-1)
        # Augment features with memory blocks weighted by attention
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block)                   #### feature_aug B,T,D
        return temporal_att, augment



def CLASM(logits, labels, lengths, device):
    """
    Custom loss function with attention-based MIL (Multiple Instance Learning).
    Args:
        logits: Logits tensor (batch_size, sequence_length)
        labels: Ground truth labels (batch_size, num_classes)
        lengths: Lengths of sequences (batch_size)
        device: Torch device
    Returns:
        loss value
    """
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        # Select top-k logits according to lengths and average them
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    """
    Binary classification loss with sigmoid activations and MIL.
    Args:
        logits: Raw logits (batch_size, sequence_length)
        labels: Ground truth binary labels (batch_size)
        lengths: Lengths of sequences (batch_size)
        device: Torch device
    Returns:
        loss value
    """
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def norm(data):
    """
    Normalize tensor by its L2 norm along last dimension.
    """
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)


def _reparameterize(mu, logvar):
    """
    Reparameterization trick for sampling from a Gaussian distribution.
    Args:
        mu: Mean tensor
        logvar: Log-variance tensor
    Returns:
        Sampled tensor
    """
    std = torch.exp(logvar).sqrt()
    epsilon = torch.randn_like(std)
    return mu + epsilon * std

def latent_loss(mu, var):
    """
    KL divergence loss for latent variables.
    """
    kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
    return kl_loss
    
class LayerNorm(nn.LayerNorm):
    """
    Wrapper for LayerNorm that preserves input dtype.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    Fast approximation of GELU activation.
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Residual attention block with multi-head self-attention and MLP.
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        Apply multi-head attention with optional padding and causal mask.
        """
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    """
    Transformer model consisting of stacked ResidualAttentionBlocks.
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    """
    Main model class combining CLIP, Graph Convolutions, Transformer, and Memory Units.
    """
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()
        self.device = 'cpu'
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        # Forecasting parameters
        self.forecast_input = 64
        self.forecast_output = 5

        # Frame predictor module (crossformer)
        self.frame_predictor = Crossformer(visual_width, self.forecast_input, self.forecast_output, self.forecast_input, 1).to(self.device)

        # Temporal transformer with attention mask
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        # Graph convolution layers for spatio-temporal modeling
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)
        
        # CLIP model loading (with frozen weights)
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

        self.encoder_mu = nn.Sequential(nn.Linear(self.visual_width, self.visual_width))
        self.encoder_var = nn.Sequential(nn.Linear(self.visual_width, self.visual_width))
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.b = 1
        self.flag = 'Train'

    def initialize_parameters(self):
        """
        Build causal attention mask with a sliding window constraint.
        """
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        """
        Create a causal attention mask with local attention windows for vision tokens.
        - Vision tokens within the same window have full attention (mask = 0).
        - Tokens outside windows are masked with -inf to block attention.
        - PyTorch expects additive attention masks where -inf prevents attention.
        
        Args:
            attn_window (int): Size of each local attention window.
        
        Returns:
            mask (Tensor): A square mask tensor of shape (visual_length, visual_length)
        """
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        # For each window segment, allow full attention (mask=0) within that segment
        for i in range(int(self.visual_length / attn_window)):
            # For last segment, might be smaller than attn_window
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        """
        Compute adjacency matrices for a batch of feature sequences based on cosine similarity.
        - Computes pairwise cosine similarity between temporal features.
        - Applies thresholding to remove weak connections (< 0.7).
        - Applies softmax normalization along dimension 1 for probabilistic adjacency.
        - Handles variable sequence lengths if seq_len provided.

        Args:
            x (Tensor): Input feature tensor of shape (B, T, D)
            seq_len (list or None): List of valid sequence lengths for each batch item
        
        Returns:
            output (Tensor): Normalized adjacency matrices (B, T, T)
        """
        soft = nn.Softmax(1)

        # Compute raw similarity: batch matrix multiplication between x and x^T
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
         
        # Normalize by product of vector norms for cosine similarity
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        """
        Encode a batch of video frames into learned visual features.
        - Adds learnable positional embeddings to each frame.
        - Processes frames through temporal transformer.
        - Computes adjacency matrices and applies graph convolutions.
        - Combines outputs and projects with a linear layer.
        
        Args:
            images (Tensor): Input video frames (B, T, visual_width)
            padding_mask (Tensor): Padding mask for frames (unused in snippet)
            lengths (list): Valid sequence lengths
        
        Returns:
            x (Tensor): Encoded visual features (B, T, output_dim)
        """
        images = images.to(torch.float)
        # Create position ids and obtain position embeddings
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        # Permute for transformer input shape (T, B, D)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings
        # Temporal transformer encoding
        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)
        # Compute adjacency matrix from temporal features
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        # Apply two separate graph convolution streams with GELU activation
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))
        # Concatenate graph conv outputs and apply final linear projection              
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        """
        Encode textual prompts into embedding space using CLIP tokenizer and model.
        - Tokenizes each prompt using CLIP tokenizer.
        - Embeds tokens using pretrained CLIP text encoder.
        - Applies positional embeddings for prompt prefix and postfix.
        
        Args:
            text (list of str): Batch of text prompts.
        
        Returns:
            text_features (Tensor): Textual embeddings aligned with visual features.
        """
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features



    def mem_unit(self, anomaly_features, normal_features):
        """
        Memory unit for anomaly detection loss calculation.
        - Uses separate memory modules for anomaly and normal features.
        - Computes triplet margin loss between normal anchors, positive anomalies, and negative anomalies.
        - Applies KL divergence loss on encoded memory latent distributions.
        - Combines BCE losses on memory attention scores with triplet and KL losses.
        
        Args:
            anomaly_features (Tensor): Features from anomaly samples.
            normal_features (Tensor): Features from normal samples.
        
        Returns:
            cost (Tensor): Combined loss for training memory units.
        """
        anomaly_features = anomaly_features.to(self.device)
        normal_features = normal_features.to(self.device)
         # Memory attention and augmentation for anomaly and normal features
        A_att, A_aug = self.a_memory(anomaly_features) 
        N_Aatt, N_Aaug = self.n_memory(anomaly_features)
            
        A_Natt, A_Naug = self.a_memory(normal_features) ###bt,btd,   normal video --->>>>> Anomaly memeory   all 0 [0,0,0,0,0,...,0]
        N_att, N_aug = self.n_memory(normal_features)   ###bt,btd,   normal video --->>>>> Normal memeory    all 1 [1,1,1,1,1,...,1]
        # Select top attentive feature indices for triplet loss sampling
        _, A_index = torch.topk(A_att, self.t//16 + 1, dim=-1)
        negative_ax = torch.gather(anomaly_features, 1, A_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)
            
        _, N_index = torch.topk(N_att, self.t//16 + 1, dim=-1)
        anchor_nx=torch.gather(normal_features, 1, N_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)

        _, P_index = torch.topk(N_Aatt, self.t//16 + 1, dim=-1)
        positivte_nx = torch.gather(anomaly_features, 1, P_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)
        # Triplet margin loss between normal anchor, positive anomaly, and negative anomaly
        triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

        N_aug_mu = self.encoder_mu(N_aug)
        N_aug_var = self.encoder_var(N_aug)
        N_aug_new = _reparameterize(N_aug_mu, N_aug_var)
            
        anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)

        A_aug_new = self.encoder_mu(A_aug)
        negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)
            
        kl_loss = latent_loss(N_aug_mu, N_aug_var)
        # Additional anomaly detection loss components
        panomaly = torch.topk(1 - N_Aatt, self.t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((self.b)).to(self.device))
        # Distance regularization to separate anomaly and normal latent representations
        distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()

        # Binary cross entropy losses for attention scores
        A_att = torch.topk(A_att, self.t//16 + 1, dim = -1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((self.b)).to(self.device))

        N_loss = self.bce(N_att, torch.ones_like((N_att)).to(self.device))    
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).to(self.device))
        # Weighted sum of all loss components
        cost = 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet_margin_loss + 0.01 * kl_loss + 0.01 * distance
        return cost# , [A_aug + N_aug]


    def forward(self, visual, padding_mask, text, lengths):
        """
        Forward pass of the model with optional future frame prediction and multi-modal embedding.
        - Predicts future frames if in 'Test' or 'Train' mode.
        - Encodes predicted visual features.
        - Computes memory losses and triplet loss during training.
        - Encodes text prompts and combines visual and text features with attention.
        - Returns raw text features, visual logits, text-visual similarity logits, and prediction loss.
        
        Args:
            visual (Tensor): Visual input frames (B, T, visual_width)
            padding_mask (Tensor): Padding mask for inputs
            text (list): List of text prompts
            lengths (list): Sequence lengths
        
        Returns:
            text_features_ori (Tensor): Original text feature embeddings
            logits1 (Tensor): Visual logits from classifier
            logits2 (Tensor): Text-visual similarity logits
            future_pred_loss (Tensor): Loss for future frame prediction
        """
        if self.flag == 'Test':
        # Flatten visual features for frame prediction
            concat_visual = visual.view(-1, self.visual_width)
            new_visual = concat_visual[:self.forecast_input]
            for i in range(self.forecast_input, concat_visual.shape[0]-self.forecast_output+1, self.forecast_output):
                pred_tensor = self.frame_predictor(torch.unsqueeze(concat_visual[i - self.forecast_input: i].to(torch.float32), 0))
                new_visual = torch.cat([new_visual, torch.squeeze(pred_tensor).to(torch.float32)], dim=0)

            new_visual = torch.cat([new_visual, concat_visual[new_visual.shape[0]:]], dim=0)
            new_visual = new_visual.view(-1, self.forecast_input, self.visual_width)

        else:
        # Predict future frames in training or other modes
            new_visual = visual[:,:self.forecast_input]
            for i in range(self.forecast_input, self.visual_length-self.forecast_output+1, self.forecast_output):
                pred_tensor = self.frame_predictor(visual[:,i - self.forecast_input: i].to(torch.float32))
                new_visual = torch.concat([new_visual, pred_tensor.to(torch.float32)], dim=1)
            new_visual = torch.concat([new_visual, visual[:,new_visual.shape[1]:]], dim=1)
        future_pred_loss = self.mse(visual, new_visual)
        visual_features = self.encode_video(new_visual, padding_mask, lengths)
        visual_features_len = len(visual_features)

        if self.flag == 'Train':
            normal_features = visual_features[:visual_features_len//2]
            anomaly_features = visual_features[visual_features_len//2:]
            A_att, A_aug = self.a_memory(anomaly_features) 
            N_Aatt, N_Aaug = self.n_memory(anomaly_features)
                
            A_Natt, A_Naug = self.a_memory(normal_features) ###bt,btd,   normal video --->>>>> Anomaly memeory   all 0 [0,0,0,0,0,...,0]
            N_att, N_aug = self.n_memory(normal_features)   ###bt,btd,   normal video --->>>>> Normal memeory    all 1 [1,1,1,1,1,...,1]

            _, A_index = torch.topk(A_att, self.t//16 + 1, dim=-1)
            negative_ax = torch.gather(anomaly_features, 1, A_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)
                
            _, N_index = torch.topk(N_att, self.t//16 + 1, dim=-1)
            anchor_nx=torch.gather(normal_features, 1, N_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)

            _, P_index = torch.topk(N_Aatt, self.t//16 + 1, dim=-1)
            positivte_nx = torch.gather(anomaly_features, 1, P_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)
                
            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = _reparameterize(N_aug_mu, N_aug_var)
                
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, self.d])).mean(1).reshape(self.b//2,self.n,-1).mean(1)
                
            kl_loss = latent_loss(N_aug_mu, N_aug_var)
                
            
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()

                
            A_att = torch.topk(A_att, self.t//16 + 1, dim = -1)[0].mean(-1)
            A_loss = self.bce(A_att, torch.ones((self.b)).to(self.device))

            N_loss = self.bce(N_att, torch.ones_like((N_att)).to(self.device))    
            A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).to(self.device))

            cost = 0.1 * (A_loss + N_loss + A_Nloss) + 0.1 * triplet_margin_loss + 0.01 * kl_loss + 0.01 * distance
            

        else:
            normal_features = visual_features[:visual_features_len//2]
            anomaly_features = visual_features[visual_features_len//2:]
            A_att, A_aug = self.a_memory(anomaly_features) 
            N_att, N_aug = self.n_memory(normal_features)  
            cost = 0
        aug_vis_features = torch.cat([visual_features, torch.cat([N_aug , A_aug])], dim=-1)
        logits1 = self.classifier(aug_vis_features + self.mlp2(aug_vis_features))
        logits1 = self.classifier(visual_features)

        text_features_ori = self.encode_textprompt(text)
        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)

        
        segment_length = logits_attn.shape[-1] // 3
        scale_first = 1.2  # Adjust this to increase the first third values
        scale_middle = 1.1  # Adjust this to decrease the middle values if needed
        scale_final = 0.78  # Adjust this to decrease the final values if needed

        # Apply scaling factors to the respective segments
        logits_attn[:segment_length] *= scale_first
        logits_attn[segment_length:2*segment_length] *= scale_middle
        logits_attn[2*segment_length:] *= scale_final

        
        visual_attn = logits_attn @ visual_features
        
        segment_length = visual_attn.shape[-1] // 3

        # Apply scaling factors to the respective segments
        visual_attn[:segment_length] *= scale_first
        visual_attn[segment_length:2*segment_length] *= scale_middle
        visual_attn[2*segment_length:] *= scale_final
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)

        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2, future_pred_loss + cost
    