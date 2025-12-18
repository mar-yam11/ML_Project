import os
import cv2
import joblib
import numpy as np

from src.features.extract_features import extract_features
from src.models.predict_with_unknown import svm_predict as svm_predict_with_unknown


def predict(dataFilePath, bestModelPath):
    predictions = []

    # load trained model
    model = joblib.load(bestModelPath)

    for img_name in os.listdir(dataFilePath):
        img_path = os.path.join(dataFilePath, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # extract features
        features = extract_features(img)

        # predict class (with unknown)
        class_id = svm_predict_with_unknown(model, features)

        predictions.append(class_id)

    return predictions
