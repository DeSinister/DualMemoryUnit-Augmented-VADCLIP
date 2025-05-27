import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('png', 'jpg', 'jpeg')):  # Ensure it is an image file
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, filenames

# Function to extract features using CLIP
def extract_features(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.cpu().numpy()

# Define the main function
def main(input_base_folder, output_base_folder, chunk_size=30, frequency=30):
    # Load the CLIP model and processor
    device = "cuda" #if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Iterate through each subfolder in the input base folder
    for video_folder in tqdm(os.listdir(input_base_folder)[36:][::-1]):
        input_folder = os.path.join(input_base_folder, video_folder)
        if os.path.isdir(input_folder):
            output_folder = os.path.join(output_base_folder, video_folder)
            os.makedirs(output_folder, exist_ok=True)
            print(f"Processing folder: {input_folder}")

            # Load images
            images, filenames = load_images_from_folder(input_folder)

            # Process images in chunks
            for i in range(0, len(images) - chunk_size + 1, frequency):
                chunk = images[i:i+chunk_size]

                # Extract features
                features = extract_features(chunk, model, processor, device)

                # Save features as .npy files
                chunk_filename = f"features_chunk_{i//frequency}.npy"
                output_file = os.path.join(output_folder, chunk_filename)
                np.save(output_file, features)
                # print(f"Saved features to {output_file} with shape {features.shape}")

if __name__ == "__main__":
    input_base_folder = "/ingenuity_NAS/dataset/public/UCFCrime/VideoFrames/Training_Normal_Videos_Anomaly"  # Replace with your input base folder path
    output_base_folder = '/ingenuity_NAS/dataset/public/UCFCrime/CLIPFeats/Training_Normal_Videos_Anomaly'  # Replace with your output base folder path
    main(input_base_folder, output_base_folder)

