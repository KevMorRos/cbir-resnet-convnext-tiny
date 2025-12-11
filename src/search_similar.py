import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from utils import (
    load_image,
    preprocess_image,
    load_npy,
    cosine_similarity,
    DEVICE
)

# Load the correct backbone
def load_backbone(backbone_name):
    if backbone_name == "resnet":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        feature_dim = 2048

    elif backbone_name == "convnext":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        model.classifier[2] = torch.nn.Identity()
        feature_dim = 768

    else:
        raise ValueError("Backbone must be 'resnet' or 'convnext'.")

    model = model.to(DEVICE)
    model.eval()
    return model, feature_dim


# Compute deep feature for a single query image
def extract_query_feature(model, img_path, feature_dim):
    img = load_image(img_path)
    if img is None:
        raise ValueError(f"Could not load query image: {img_path}")

    img_tensor = preprocess_image(img)

    with torch.no_grad():
        vec = model(img_tensor).cpu().numpy().flatten()

    assert vec.shape[0] == feature_dim
    return vec

# Compute cosine similarities between query and dataset
def compute_similarities(query_vec, db_features):
    similarities = db_features @ query_vec / (
        np.linalg.norm(db_features, axis=1) * np.linalg.norm(query_vec)
    )
    return similarities


# Visualization: Create a retrieval result figure
def save_retrieval_plot(query_path, retrieved_paths, save_path):
    num_results = len(retrieved_paths)
    fig, ax = plt.subplots(1, num_results + 1, figsize=(15, 4))

    # Query image
    query_img = Image.open(query_path).convert('RGB')
    ax[0].imshow(query_img)
    ax[0].set_title("Query")
    ax[0].axis("off")

    # Retrieved images
    for i, img_path in enumerate(retrieved_paths):
        img = Image.open(img_path).convert('RGB')
        ax[i + 1].imshow(img)
        ax[i + 1].set_title(f"Rank {i+1}")
        ax[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Main search function
def search(query_path, backbone, k=5):
    print(f"\n Running CBIR using backbone: {backbone.upper()}")

    # Load backbone model
    model, feature_dim = load_backbone(backbone)

    # Load stored dataset features
    feature_file = f"features/{backbone}_features.npy"
    db_features = load_npy(feature_file)
    filenames = load_npy("features/filenames.npy")

    print(f"Loaded {db_features.shape[0]} feature vectors.")

    # Compute query feature
    print("Extracting query feature...")
    query_vec = extract_query_feature(model, query_path, feature_dim)

    # Compute similarities
    similarities = compute_similarities(query_vec, db_features)

    # Get top-k results
    top_k_idx = np.argsort(similarities)[::-1][:k]
    retrieved_paths = filenames[top_k_idx]

    print("\nTop retrieved images:")
    for i, p in enumerate(retrieved_paths):
        print(f"{i+1}. {p}")

    # Save retrieval visualization
    save_path = f"output/retrieval_{backbone}.png"
    save_retrieval_plot(query_path, retrieved_paths, save_path)

    print(f"\n Retrieval plot saved at: {save_path}")

# Script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Path to query image.")
    parser.add_argument("--backbone", type=str, choices=["resnet", "convnext"], default="resnet")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    search(args.query, args.backbone, k=args.k)
