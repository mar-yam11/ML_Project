import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model

# load pretrained CNN once
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

def extract_features(image):
    # resize to CNN input size
    image = cv2.resize(image, (224, 224))

    # convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # prepare image for CNN
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # extract features
    features = base_model.predict(image, verbose=0)

    return features.flatten()
