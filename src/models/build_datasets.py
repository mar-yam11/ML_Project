import os
import cv2
import numpy as np
from ..features.extract_features import extract_features

data_path = "data/augmented"
classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

X = []
y = []

for label in range(len(classes)):
    folder = os.path.join(data_path, classes[label])

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        features = extract_features(img)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

np.save("src/models/X.npy", X)
np.save("src/models/y.npy", y)

print("X shape:", X.shape)
print("y shape:", y.shape)
