import os
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights

from utils import (
    load_image,
    preprocess_image,
    list_images,
    save_npy,
    ensure_dir,
    DEVICE,
    progress,
)

#  Feature extraction with ResNet-50
def extract_features(dataset_path, output_dir="features/"):
    ensure_dir(output_dir)

    # Load all image paths + labels
    print(" Scanning dataset...")
    image_paths, labels = list_images(dataset_path)
    num_images = len(image_paths)
    print(f"Found {num_images} images.")

    # Load pretrained ResNet-50
    print(" Loading ResNet-50 (ImageNet pretrained)...")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Remove classifier head → output 2048-d features
    model.fc = torch.nn.Identity()

    # Use evaluation mode
    model = model.to(DEVICE)
    model.eval()

    # Allocate feature matrix
    features = np.zeros((num_images, 2048), dtype=np.float32)

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
    save_npy(os.path.join(output_dir, "resnet_features.npy"), features)
    save_npy(os.path.join(output_dir, "filenames.npy"), np.array(image_paths))
    save_npy(os.path.join(output_dir, "labels.npy"), np.array(labels))

    print(" Feature extraction complete!")
    print(f"Saved to folder: {output_dir}")


if __name__ == "__main__":
    DATA_PATH = "data/256_ObjectCategories/"
    extract_features(DATA_PATH)