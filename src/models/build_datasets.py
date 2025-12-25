# Import OS utilities for file path handling
import os
# Import OpenCV for reading images
import cv2
# Import NumPy for arrays and saving .npy files
import numpy as np
# Import our feature extractor
from ..features.extract_features import extract_features

 # Root folder containing augmented training images
data_path = "data/augmented"
 # Class folder names (labels correspond to their index)
classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

 # List to collect feature vectors
X = []
 # List to collect label IDs
y = []

 # Loop over each class index
for label in range(len(classes)):
    # Build path to the class folder
    folder = os.path.join(data_path, classes[label])

    # Iterate over all image files in the class folder
    for img_name in os.listdir(folder):
        # Build full path to the image
        img_path = os.path.join(folder, img_name)
        # Read the image as BGR
        img = cv2.imread(img_path)

        # Skip files that could not be read
        if img is None:
            continue

        # Compute feature vector for the image
        features = extract_features(img)
        # Store features and corresponding label
        X.append(features)
        y.append(label)

 # Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

 # Save dataset arrays to disk for training
np.save("src/models/X.npy", X)
np.save("src/models/y.npy", y)

 # Print shapes for a quick sanity check
print("X shape:", X.shape)
print("y shape:", y.shape)
