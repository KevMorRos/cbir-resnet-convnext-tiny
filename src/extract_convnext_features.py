import os
import numpy as np
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from utils import (
    load_image,
    preprocess_image,
    list_images,
    save_npy,
    ensure_dir,
    DEVICE,
    progress
)


#  Feature extraction with ConvNeXt-Tiny
def extract_features(dataset_path, output_dir="features/"):
    ensure_dir(output_dir)

    # Load all image paths + labels
    print("📂 Scanning dataset...")
    image_paths, labels = list_images(dataset_path)
    num_images = len(image_paths)
    print(f"Found {num_images} images.")

    # Load pretrained ConvNeXt-Tiny
    print("📦 Loading ConvNeXt-Tiny (ImageNet pretrained)...")
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)

    # Replace classifier with Identity to get raw embeddings
    model.classifier[2] = torch.nn.Identity()   # classifier = [LayerNorm, Flatten, Linear]
    
    # Move to device
    model = model.to(DEVICE)
    model.eval()

    # Output embedding dimension is 768
    features = np.zeros((num_images, 768), dtype=np.float32)

    # Extract features per image
    for i, img_path in progress(enumerate(image_paths), desc="Extracting", total=num_images):
        img = load_image(img_path)
        if img is None:
            continue

        img_tensor = preprocess_image(img)

        with torch.no_grad():
            vec = model(img_tensor).cpu().numpy().flatten()
            features[i] = vec

    # Save everything
    print(" Saving feature files...")
    save_npy(os.path.join(output_dir, "convnext_features.npy"), features)

    # filenames.npy and labels.npy were saved by ResNet script
    print(" ConvNeXt feature extraction complete!")
    print(f"Saved to folder: {output_dir}")


if __name__ == "__main__":
    DATA_PATH = "data/256_ObjectCategories/" 
    extract_features(DATA_PATH)
