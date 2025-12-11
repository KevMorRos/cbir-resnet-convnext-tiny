import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def get_device():
    """
    Returns MPS if available (Mac M1), otherwise CUDA or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_device()

#  Directory Helpers
def ensure_dir(path):
    """
    Creates directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


#  Image Loading & Preprocessing
def load_image(path):
    """
    Loads an image from file and converts it to RGB.
    """
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def get_transform(image_size=224):
    """
    Standard preprocessing transform: resize -> tensor -> normalize.
    Works for ResNet and ConvNeXt models (ImageNet normalization).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   
            std=[0.229, 0.224, 0.225]    
        )
    ])


def preprocess_image(img, image_size=224):
    """
    Returns preprocessed tensor (1, C, H, W).
    """
    transform = get_transform(image_size)
    tensor = transform(img).unsqueeze(0)
    return tensor.to(DEVICE)


def list_images(dataset_root):
    """
    Returns 2 lists:
    - image file paths
    - labels (folder names)
    
    Example:
    data/Caltech256/010.penguin/abc.jpg -> label '010.penguin'
    """
    image_paths = []
    labels = []

    classes = sorted(os.listdir(dataset_root))

    for cls in classes:
        class_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(class_dir, filename)
                image_paths.append(full_path)
                labels.append(cls)

    return image_paths, labels

#  Similarity Measures
def cosine_similarity(a, b):
    """
    Computes cosine similarity between two vectors.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def compute_top_k_similarities(query_vec, db_vectors, k=5):
    """
    Given a query vector and a matrix of feature vectors,
    returns indices of top-k most similar images.
    """
    similarities = db_vectors @ query_vec / (
        np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vec)
    )

    top_k_idx = np.argsort(similarities)[::-1][:k]
    return top_k_idx, similarities[top_k_idx]

#  Saving / Loading Features
def save_npy(path, arr):
    """
    Saves numpy array to .npy
    """
    np.save(path, arr)


def load_npy(path):
    """
    Loads numpy array from .npy
    """
    return np.load(path)


#  Progress Bar Helper
def progress(iterable, desc="Processing", total=None):
    """
    Wrapper around tqdm for clean progress bars.
    """
    return tqdm(iterable, desc=desc, total=total, ncols=80)
