import argparse
import numpy as np
from tqdm import tqdm

from utils import load_npy

# Compute cosine similarity
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Compute Precision@K for classical features
def evaluate_classical(features, filenames, labels, K_vals=[5, 10], num_queries=200):
    total_images = features.shape[0]

    # Randomly select query indices
    np.random.seed(0)
    query_indices = np.random.choice(total_images, size=num_queries, replace=False)

    correct_at_k = {k: 0 for k in K_vals}

    for idx in tqdm(query_indices, desc="Evaluating Classical Features"):
        query_vec = features[idx]
        query_label = labels[idx]

        # Compute similarities
        sims = features @ query_vec / (
            np.linalg.norm(features, axis=1) * np.linalg.norm(query_vec)
        )

        # Sort similarities and ignore self-match
        ranked_idx = np.argsort(sims)[::-1][1:max(K_vals)+1]
        ranked_labels = labels[ranked_idx]

        # Precision@K
        for k in K_vals:
            top_k = ranked_labels[:k]
            correct = np.sum(top_k == query_label)
            correct_at_k[k] += correct / k

    # Return precisions
    return {k: correct_at_k[k] / num_queries for k in K_vals}


# Script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["color", "lbp"],
        required=True,
        help="Choose between color or lbp."
    )
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()

    # Load classical feature vectors
    if args.feature_type == "color":
        features = load_npy("features/color_features.npy")
    else:
        features = load_npy("features/lbp_features.npy")

    # Load filenames + labels (same as deep features)
    filenames = load_npy("features/filenames.npy")
    labels = load_npy("features/labels.npy")

    print(f"Loaded {features.shape[0]} {args.feature_type.upper()} feature vectors.")

    # Evaluate
    results = evaluate_classical(
        features=features,
        filenames=filenames,
        labels=labels,
        num_queries=args.queries
    )

    # Print precision scores
    print("\nEvaluation Results for", args.feature_type.upper())
    for k, p in results.items():
        print(f"Precision@{k}: {p:.4f}")
