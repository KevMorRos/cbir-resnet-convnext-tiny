import argparse
import numpy as np
from utils import load_npy
from tqdm import tqdm

# Compute precision@k
def precision_at_k(pred_labels, true_label, k):
    return np.sum(pred_labels[:k] == true_label) / k

# Evaluate deep backbones
def evaluate_backbone(backbone, queries=200):
    features = load_npy(f"features/{backbone}_features.npy")
    filenames = load_npy("features/filenames.npy")
    labels = load_npy("features/labels.npy")

    print(f"\nEvaluating backbone: {backbone.upper()}")
    print(f"Using {queries} random queries...\n")

    idx = np.random.choice(len(features), size=queries, replace=False)

    p5_list, p10_list = [], []

    for i in tqdm(idx):
        qvec = features[i]
        sims = features @ qvec / (
            np.linalg.norm(features, axis=1) * np.linalg.norm(qvec)
        )
        ranked = np.argsort(sims)[::-1]
        ranked = ranked[ranked != i]  # remove self-match

        pred_labels = labels[ranked]

        p5 = precision_at_k(pred_labels, labels[i], 5)
        p10 = precision_at_k(pred_labels, labels[i], 10)

        p5_list.append(p5)
        p10_list.append(p10)

    print(f"Precision@5:  {np.mean(p5_list):.4f}")
    print(f"Precision@10: {np.mean(p10_list):.4f}")

# Evaluate classical features
def evaluate_classical():
    color = load_npy("features/color_features.npy")
    lbp = load_npy("features/lbp_features.npy")
    labels = load_npy("features/labels.npy")

    print("\nEvaluating CLASSICAL features...\n")

    # COLOR
    print("COLOR HISTOGRAMS:")
    p5_list, p10_list = [], []

    for i in range(len(color)):
        qvec = color[i]
        sims = color @ qvec / (
            np.linalg.norm(color, axis=1) * np.linalg.norm(qvec)
        )
        ranked = np.argsort(sims)[::-1]
        ranked = ranked[ranked != i]

        pred_labels = labels[ranked]
        p5 = precision_at_k(pred_labels, labels[i], 5)
        p10 = precision_at_k(pred_labels, labels[i], 10)

        p5_list.append(p5)
        p10_list.append(p10)

    print(f"  Precision@5:  {np.mean(p5_list):.4f}")
    print(f"  Precision@10: {np.mean(p10_list):.4f}\n")

    # LBP
    print("LBP TEXTURE:")
    p5_list, p10_list = [], []

    for i in range(len(lbp)):
        qvec = lbp[i]
        sims = lbp @ qvec / (
            np.linalg.norm(lbp, axis=1) * np.linalg.norm(qvec)
        )
        ranked = np.argsort(sims)[::-1]
        ranked = ranked[ranked != i]

        pred_labels = labels[ranked]
        p5 = precision_at_k(pred_labels, labels[i], 5)
        p10 = precision_at_k(pred_labels, labels[i], 10)

        p5_list.append(p5)
        p10_list.append(p10)

    print(f"  Precision@5:  {np.mean(p5_list):.4f}")
    print(f"  Precision@10: {np.mean(p10_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["resnet", "convnext"])
    parser.add_argument("--method", type=str, choices=["classical"], help="Evaluate classical features")
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()

    if args.method == "classical":
        evaluate_classical()
    else:
        evaluate_backbone(args.backbone, queries=args.queries)
