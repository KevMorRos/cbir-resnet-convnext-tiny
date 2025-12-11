# Content-Based Image Retrieval (CBIR) System  
### Using ResNet-50, ConvNeXt-Tiny, and Classical Features  
**CS4337 – Introduction to Computer Vision**  
**Author:** Kevin Morales  

---

## Overview

This project implements a full **Content-Based Image Retrieval (CBIR)** system using:

### **Deep Feature Extractors**
- **ResNet-50** (ImageNet pretrained)
- **ConvNeXt-Tiny** (ImageNet pretrained)

### **Classical Features**
- Color Histograms  
- Local Binary Patterns (LBP)

The system retrieves the **Top-K most visually similar images** from the **Caltech-256** dataset and supports:

- Quantitative evaluation (Precision@5 / Precision@10)  
- Dataset-based success & failure examples  
- Retrieval on **external real-world images**  
- Retrieval on **unseen categories** (e.g., microphone)

---

# ⚙️ Installation Instructions

This project was done in a conda virtual enviroment.

---

# **After Installing Conda (Recommended)**

### 1. Create a Conda environment:
```
conda create -n cbir python=3.10
conda activate cbir
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Download Caltech-256 Dataset
Download [Caltech-256](https://data.caltech.edu/records/nyy15-4j048)
Move contents into data directory.

### Feature Extraction

Feature extraction converts images into numerical vectors representing visual patterns.
These vectors are stored as .npy files for fast search.

1. Extract ResNet-50 Features
```
python src/extract_resnet_features.py
```
### Creates:
```
features/resnet_features.npy
features/filenames.npy
features/labels.npy
```

2. Extract ConvNext-Tiny Features
```
python src/extract_convnext_features.py
```
### Creates:
```
features/convnext_features.npy
```
3. Extract Classical Features (Color + LBP)
```
python src/extract_classical_features.py
```
### Creates: 
```
features/color_features.npy
features/lbp_features.npy
```

## Quantitative Evaluation
Evaluate retrieval accuracy with 200 random queries.

4. ResNet-50 Evaluation
```
python src/evaluate_precision.py --backbone resnet
```
### Ouput Looks Like:
```
Precision@5: 0.7590
Precision@10: 0.7410
```
5. ConvNeXt-Tiny Evaluation 
```
python src/evaluate_precision.py --backbone convnext
```
### Output Looks Like
```
Precision@5:  0.8360
Precision@10: 0.8095
```
6. Classical Feature Evaluation
```
python src/evaluate_classical_features.py --feature_type color --queries 200
python src/evaluate_classical_features.py --feature_type lbp --queries 200
```
7. ResNet50 Visualization
- Select 50 random images from dataset
- Retrieves Top-5 most similar images
- Saves success/failure retrievelgrids
```
output/success/query_XXX.png
output/failure/query_YYY.png
```
8. ConvNeXt-Tiny Visualization
```
python src/visualize_results.py --backbone convnext
```
9. External Query Retrieval
- Place any .jpg/.png images into: 
```
data/external_queries/
```
10. Run Retrieval: 
- Loads each external image
- Extracts ConvNeXt features
- Finds top-5 most similar images
- Saves visual retrieval grids
```
python src/batch_search.py --query-dir data/external_queries --backbone convnext
```
### Creates: 
```
output/external_queries/convnext/<filename>_retrieval.png
```
11. Single Image Search Example
```
python src/batch_search.py --query path/to/image.jpg --backbone convnext
```
### Creates:
```
output/external_queries/<image>_retrieval.png
```
