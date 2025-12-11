📂 Data Processing Pipeline - Material Stream Identification
This directory contains the data preprocessing pipeline for the Material Stream Identification project. The pipeline is designed to transform raw images into numerical feature vectors suitable for Machine Learning classifiers (SVM & k-NN).

🛠️ Pipeline Overview
The pipeline consists of three sequential stages:

Data Cleaning: Removing corrupt files and standardizing image formats.

Data Augmentation & Balancing: Increasing the dataset size (≥30%) and balancing class distributions using geometric and photometric transformations.

Feature Extraction: Converting images into Histogram of Oriented Gradients (HOG) feature vectors.

🚀 Usage Instructions
Run the scripts in the following order:

Bash

# Step 1: Clean the raw data
python src/clean.py

# Step 2: Augment and balance the dataset
python src/augmentation.py

# Step 3: Extract features (HOG)
python src/extraction.py
🔍 Detailed Methodology
1. Data Cleaning (clean.py)
Input: data/raw

Output: data/cleaned

Process:

Iterates through all class folders.

Verifies image integrity using PIL.Image.verify() to detect and skip corrupt files.

Converts all images to RGB mode to handle consistent channel depth.

Saves validated images in standard JPEG format.

2. Data Augmentation & Balancing (augmentation.py)
Input: data/cleaned

Output: data/augmented

Goal: To achieve class balance (target: 600 images per class) and improve model generalization.

Techniques Used:

Geometric Transformations:

RandomRotation(30): Simulates objects dropped at different angles.

RandomHorizontalFlip & RandomVerticalFlip: Accounts for orientation invariance of waste items.

Photometric Transformations:

ColorJitter (Brightness, Contrast, Saturation, Hue): Simulates varying lighting conditions in the sorting facility.

Resizing: All images are resized to 64x64 pixels to reduce computational load.

Logic: The script first copies valid original images, then generates augmented versions until the class count reaches 600.

3. Feature Extraction (extraction.py)
Input: data/augmented

Output: data/features/data.npz

Method: Histogram of Oriented Gradients (HOG).

Preprocessing:

Images are converted to Grayscale (color is less critical for shape-based classification).

Images are resized to 64x64 (consistent with augmentation step).

HOG Parameters:

orientations: 9

pixels_per_cell: (8, 8)

cells_per_block: (2, 2)

block_norm: 'L2-Hys' (Normalization for robustness against illumination changes).

Final Output Structure: The output is a compressed NumPy file (data.npz) containing:

X: Feature matrix (Shape: [n_samples, feature_vector_size]).

y: Label vector (Shape: [n_samples]).

class_map: Dictionary mapping class indices to names (e.g., {0: 'glass', 1: 'paper'}).

📁 Directory Structure
Plaintext

Project/
├── data/
│   ├── raw/            # Original dataset
│   ├── cleaned/        # Validated images
│   ├── augmented/      # Balanced dataset (600 imgs/class)
│   └── features/       # Final .npz file
├── src/
│   ├── clean.py
│   ├── augmentation.py
│   └── extraction.py
└── README.md
