# Video Anomaly Detection

This repository contains a PyTorch implementation for **Video-Text Prompted Anomaly Detection**, which combines visual feature forecasting, GCN-based representation learning, and memory modules with CLIP-based text embeddings. The model is capable of detecting anomalies in video frames conditioned on natural language prompts.

## 🚀 Features

- Visual frame encoder with temporal attention and forecasting
- Dual memory modules for modeling normal and anomalous patterns
- Triplet loss and KL-divergence-based latent regularization
- CLIP-based text encoding for prompt-based anomaly scoring
- Multi-level visual-text alignment via dot product and attention
- Forecasting-based future prediction loss


To Extract CLIP Features
```
python extract_clip.py
```

Traing and infer for UCF-Crime dataset
```
python ucf_train.py
python ucf_test.py
```


## References
Following repos were referenced for the code.
* [UR-DMU](https://github.com/henrryzh1/UR-DMU)
* [VADCLIP](https://github.com/nwpu-zxr/VadCLIP)
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
* [DeepMIL](https://github.com/Roc-Ng/DeepMIL)
