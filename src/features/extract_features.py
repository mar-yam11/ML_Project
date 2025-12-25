'''
1. Purpose of This File

This file is responsible for feature extraction only. It converts an input image (from disk or webcam) into a fixed-length numerical feature vector using a pretrained CNN (MobileNetV2). It does not perform classification and does not train a neural network. The extracted features are later passed to classical machine learning classifiers such as SVM and kNN.
'''
# Import OpenCV for image resizing and color conversion
import cv2
# Import NumPy for array manipulation
import numpy as np
# Import MobileNetV2 model and its preprocessing function
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# Import TensorFlow backend
import tensorflow as tf
# Import Keras namespace (not strictly needed here)
from tensorflow import keras

# Initialize a placeholder Sequential model (unused placeholder)
model = tf.keras.models.Sequential()

# Create a pretrained MobileNetV2 base to extract features (loaded once)
base_model = MobileNetV2(
    # Load weights trained on ImageNet
    # Uses pretrained weights learned from over one million images.
    weights="imagenet",
    # Remove the top classification layers
    # Removes the original classification layers, keeping only the feature extractor.
    include_top=False,
    # Use global average pooling to get a fixed-length vector
    pooling="avg",
    # Expect 224x224 RGB input
    input_shape=(224, 224, 3)
)

def extract_features(image):
    '''
    The function extract_features(image) takes a single image as input and outputs a one-dimensional feature vector. This vector represents high-level visual information such as textures, shapes, and object parts learned by the CNN.
    Resizing: The image is resized to 224×224 pixels to match the CNN’s expected input size.

    Color Conversion: The image is converted from BGR (used by OpenCV) to RGB (expected by Keras).

    Batch Dimension Addition: The image shape is expanded from (224, 224, 3) to (1, 224, 224, 3) because CNNs operate on batches.

    Normalization: preprocess_input scales and normalizes pixel values exactly as MobileNetV2 was trained.
    '''
    # Resize input image to 224x224 for the CNN
    image = cv2.resize(image, (224, 224))

    # Convert color from BGR (OpenCV) to RGB (Keras expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add batch dimension and preprocess for MobileNetV2
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Extract deep features using the base model
    # The output is a tensor of shape (1, 1280), where each value represents a learned visual feature. This output is then flattened into a one-dimensional vector of length 1280, which is suitable for classical machine learning models.
    features = base_model.predict(image, verbose=0)

    # Return a 1D feature vector
    return features.flatten()
