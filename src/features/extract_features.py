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
    weights="imagenet",
    # Remove the top classification layers
    include_top=False,
    # Use global average pooling to get a fixed-length vector
    pooling="avg",
    # Expect 224x224 RGB input
    input_shape=(224, 224, 3)
)

def extract_features(image):
    # Resize input image to 224x224 for the CNN
    image = cv2.resize(image, (224, 224))

    # Convert color from BGR (OpenCV) to RGB (Keras expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add batch dimension and preprocess for MobileNetV2
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Extract deep features using the base model
    features = base_model.predict(image, verbose=0)

    # Return a 1D feature vector
    return features.flatten()
