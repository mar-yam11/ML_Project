# Import OpenCV library for camera capture and display
import cv2
# Import joblib for loading saved ML models
import joblib
# Import our feature extractor function
from ..features.extract_features import extract_features
# Import SVM prediction helper that handles unknown class
from ..models.predict_with_unknown import svm_predict as svm_predict_with_unknown

# Load pre-trained SVM model from disk
model = joblib.load("src/models/saved_models/svm_model.joblib")

# List of class names corresponding to model outputs
classes = [
    "Glass",
    "Paper",
    "Cardboard",
    "Plastic",
    "Metal",
    "Trash",
    "Unknown"
]

# Open default webcam (index 0)
cap = cv2.VideoCapture(0)

# Inform user how to quit
print("Camera started... Press q to quit")

# Main loop: read frames until user quits
while True:
    # Read a frame; ret indicates success
    ret, frame = cap.read()
    # If capture failed, exit loop
    if not ret:
        break

    # Compute CNN-based feature vector for current frame
    features = extract_features(frame)
    # Get predicted label index, with 'Unknown' handling
    label_id = svm_predict_with_unknown(model, features)
    # Map label index to class name
    label = classes[label_id]

    # Draw predicted label text on the frame
    cv2.putText(
        frame,  # image to draw the text on
        label,  # text string to render (predicted class name)
        (20, 40),  # bottom-left corner of text baseline (x, y) in pixels
        cv2.FONT_HERSHEY_SIMPLEX,  # font face used to render the text
        1,  # font scale controlling text size (multiplier)
        (0, 255, 0),  # text color in BGR format (green)
        2  # thickness of the text strokes in pixels
    )

    # Show the annotated frame in a window
    cv2.imshow("Material Classification", frame)  # window title; image data to display

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera resources
cap.release()
# Close any OpenCV windows
cv2.destroyAllWindows()