import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from utils import load_npy, load_image, preprocess_image, DEVICE


# Load backbone
def load_backbone(backbone):
    if backbone == "resnet":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        feat_dim = 2048

    elif backbone == "convnext":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        model.classifier[2] = torch.nn.Identity()
        feat_dim = 768

    else:
        raise ValueError("Backbone must be 'resnet' or 'convnext'")

    model = model.to(DEVICE)
    model.eval()
    return model, feat_dim


# Extract query feature
def extract_feature(model, img_path, feat_dim):
    img = load_image(img_path)
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        vec = model(img_tensor).cpu().numpy().flatten()
    return vec


# Compute cosine similarities
def rank_images(query_vec, db_features, top_k=5):
    sims = db_features @ query_vec / (
        np.linalg.norm(db_features, axis=1) * np.linalg.norm(query_vec)
    )
    ranked_idx = np.argsort(sims)[::-1][:top_k]
    return ranked_idx, sims[ranked_idx]


# Save retrieval visualization for each external query
def save_grid(query_path, retrieved_paths, save_path, title):
    num = len(retrieved_paths)
    fig, ax = plt.subplots(1, num + 1, figsize=(15, 4))

    # Query image
    qimg = Image.open(query_path).convert("RGB")
    ax[0].imshow(qimg)
    ax[0].set_title("Query")
    ax[0].axis("off")

    # Retrieved images
    for i, img_path in enumerate(retrieved_paths):
        img = Image.open(img_path).convert("RGB")
        ax[i + 1].imshow(img)
        ax[i + 1].set_title(f"Rank {i+1}")
        ax[i + 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Batch search function
def batch_search(query_dir, backbone, top_k=5):

    # Load backbone model
    model, feat_dim = load_backbone(backbone)

    # Load features + filenames
    feature_file = f"features/{backbone}_features.npy"
    db_features = load_npy(feature_file)
    filenames = load_npy("features/filenames.npy")

    # Output folder
    out_dir = f"output/external_queries/{backbone}/"
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over external query images
    for fname in os.listdir(query_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        query_path = os.path.join(query_dir, fname)
        print(f"\n Processing external query: {fname}")

        # Extract feature
        qvec = extract_feature(model, query_path, feat_dim)

        # Rank DB images
        ranked_idx, sims = rank_images(qvec, db_features, top_k)

        retrieved_paths = filenames[ranked_idx]

        # Save visualization
        save_img_path = os.path.join(out_dir, f"{fname}_retrieval.png")
        save_grid(query_path, retrieved_paths, save_img_path,
                  title=f"{backbone.upper()} Retrieval for {fname}")

        print(f"Saved: {save_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-dir", type=str, required=True,
                        help="Folder with external query images.")
    parser.add_argument("--backbone", type=str, choices=["resnet", "convnext"],
                        required=True, help="Feature extraction model.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    batch_search(args.query_dir, args.backbone, top_k=args.k)
