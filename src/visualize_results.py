import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from utils import load_npy, load_image, preprocess_image, DEVICE

# Load backbone models
def load_backbone(backbone):
    if backbone == "resnet":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        feature_dim = 2048

    elif backbone == "convnext":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        model.classifier[2] = torch.nn.Identity()
        feature_dim = 768

    else:
        raise ValueError("Backbone must be 'resnet' or 'convnext'.")

    model = model.to(DEVICE)
    model.eval()
    return model, feature_dim

# Extract feature of one query image
def extract_feature(model, img_path, feature_dim):
    img = load_image(img_path)
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        vec = model(img_tensor).cpu().numpy().flatten()
    return vec


# Compute similarity ranking
def rank_images(query_vec, db_features, top_k=5):
    sims = db_features @ query_vec / (
        np.linalg.norm(db_features, axis=1) * np.linalg.norm(query_vec)
    )
    ranked_idx = np.argsort(sims)[::-1]
    return ranked_idx[:top_k], sims[ranked_idx[:top_k]]

# Save visualization grid
def save_grid(query_path, result_paths, save_path, title):
    num = len(result_paths)
    fig, ax = plt.subplots(1, num + 1, figsize=(15, 4))

    # Query
    qimg = Image.open(query_path).convert("RGB")
    ax[0].imshow(qimg)
    ax[0].set_title("Query")
    ax[0].axis("off")

    # Retrieved images
    for i, img_path in enumerate(result_paths):
        img = Image.open(img_path).convert("RGB")
        ax[i + 1].imshow(img)
        ax[i + 1].set_title(f"Rank {i+1}")
        ax[i + 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Generate success & failure visualizations
def visualize(backbone, num_samples=50, top_k=5):
    # Load features
    feature_file = f"features/{backbone}_features.npy"
    features = load_npy(feature_file)
    filenames = load_npy("features/filenames.npy")
    labels = load_npy("features/labels.npy")

    # Load backbone model
    model, feature_dim = load_backbone(backbone)

    # Create output folders
    os.makedirs("output/success", exist_ok=True)
    os.makedirs("output/failure", exist_ok=True)

    # Random queries
    np.random.seed(0)
    indices = np.random.choice(len(features), size=num_samples, replace=False)

    print(f"\nGenerating visualizations for {backbone.upper()}...")

    for idx in tqdm(indices):
        query_path = filenames[idx]
        query_label = labels[idx]

        # Extract query feature
        qvec = extract_feature(model, query_path, feature_dim)

        # Rank dataset
        ranked_idx, _ = rank_images(qvec, features, top_k=top_k+1)

        # Remove self-match if present
        ranked_idx = ranked_idx[ranked_idx != idx][:top_k]
        retrieved_paths = filenames[ranked_idx]
        retrieved_labels = labels[ranked_idx]

        # Determine success or failure
        is_success = np.any(retrieved_labels == query_label)

        if is_success:
            save_path = f"output/success/query_{idx}.png"
            title = f"SUCCESS — label: {query_label}"
        else:
            save_path = f"output/failure/query_{idx}.png"
            title = f"FAILURE — label: {query_label}"

        save_grid(query_path, retrieved_paths, save_path, title)

    print("Visualization generation complete!")
    print("Success images → output/success/")
    print("Failure images → output/failure/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet",
                        choices=["resnet", "convnext"],
                        help="Backbone model to use for visualization.")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of random dataset queries to visualize.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of retrievals to show per query.")
    args = parser.parse_args()

    visualize(args.backbone, num_samples=args.num_samples, top_k=args.top_k)