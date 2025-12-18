import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image):
    image = cv2.resize(image, (128, 128))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )

    # color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    features = np.concatenate((hog_features, hist))

    return features
