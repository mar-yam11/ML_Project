import os
import cv2
import joblib
import numpy as np
import pandas as pd  

from src.features.extract_features import extract_features
from src.models.predict_with_unknown import svm_predict as svm_predict_with_unknown


classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]  


def predict(dataFilePath, bestModelPath, output_excel="predictions.xlsx"):

    image_names = []
    predicted_classes = []

    # Load the trained model
    model = joblib.load(bestModelPath)

    print("Processing images...")

    for img_name in os.listdir(dataFilePath):
        img_path = os.path.join(dataFilePath, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_name}")
            continue

        # Extract features using MobileNetV2
        features = extract_features(img)

        # Predict class (returns 0-5 or 6 for unknown)
        class_id = svm_predict_with_unknown(model, features)

        # Convert class_id to class name
        if class_id == 6:
            predicted_class = "unknown"
        else:
            predicted_class = classes[class_id]

        # Collect results
        image_names.append(img_name)
        predicted_classes.append(predicted_class)

        print(f"{img_name} -> {predicted_class}")

    # Save results to Excel 
    if image_names:
        df = pd.DataFrame({
            "Image Name": image_names,
            "Predicted Class": predicted_classes
        })
        df.to_excel(output_excel, index=False)
        print(f"\nResults successfully saved to: {output_excel}")
        print(f"Number of processed images: {len(image_names)}")
    else:
        print("No valid images were processed.")

   
    return predicted_classes



predict("sample", "src/models/saved_models/knn_model.joblib", "results/predictions.xlsx")