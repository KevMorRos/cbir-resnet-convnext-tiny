import os
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import cv2

from utils import (
    load_image,
    list_images,
    save_npy,
    ensure_dir,
    progress
)

# LBP Parameters
LBP_POINTS = 24        # Number of sampling points
LBP_RADIUS = 3         # Radius for LBP
LBP_METHOD = "uniform" 


# Extract COLOR HISTOGRAM feature (normalized)
def extract_color_histogram(img, bins=(32, 32, 32)):
    """
    Compute a normalized color histogram in HSV space.
    Returns a 96-dim feature vector.
    """
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    hist = cv2.normalize(hist, hist).flatten()
    return hist


# Extract LBP texture feature (normalized)
def extract_lbp(img):
    """
    Compute Local Binary Pattern texture descriptor.
    Returns a 26-dim uniform LBP histogram.
    """
    img_np = np.array(img)
    gray = rgb2gray(img_np)

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)

    # Uniform LBP produces P+2 bins (26)
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, LBP_POINTS + 3),
                           range=(0, LBP_POINTS + 2))

    hist = hist.astype("float")
    hist /= hist.sum()  # normalize
    return hist


# Extract BOTH features for ALL images
def extract_features(dataset_path, output_dir="features/"):
    ensure_dir(output_dir)

    print("📂 Scanning dataset...")
    image_paths, labels = list_images(dataset_path)
    num_images = len(image_paths)
    print(f"Found {num_images} images.")

    # Pre-allocate feature arrays
    color_features = []
    lbp_features = []

    print("🎨 Extracting Color Histograms + LBP Texture Features...")

    for img_path in progress(image_paths, desc="Extracting Features"):
        img = load_image(img_path)
        if img is None:
            continue

        # Extract Color Histogram
        color_feat = extract_color_histogram(img)
        color_features.append(color_feat)

        # Extract LBP
        lbp_feat = extract_lbp(img)
        lbp_features.append(lbp_feat)

    # Convert to arrays
    color_features = np.array(color_features, dtype=np.float32)
    lbp_features = np.array(lbp_features, dtype=np.float32)

    # Save them
    print(" Saving feature files...")
    save_npy(os.path.join(output_dir, "color_features.npy"), color_features)
    save_npy(os.path.join(output_dir, "lbp_features.npy"), lbp_features)

    print(" Color + LBP feature extraction complete!")
    print(f"Saved in: {output_dir}")


if __name__ == "__main__":
    DATA_PATH = "data/256_ObjectCategories/" 
    extract_features(DATA_PATH)
