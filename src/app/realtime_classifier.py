import cv2
import joblib
from ..features.extract_features import extract_features
from ..models.predict_with_unknown import svm_predict as svm_predict_with_unknown

# load model
model = joblib.load("src/models/saved_models/svm_model.joblib")

classes = [
    "Glass",
    "Paper",
    "Cardboard",
    "Plastic",
    "Metal",
    "Trash",
    "Unknown"
]

cap = cv2.VideoCapture(0)

print("Camera started... Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = extract_features(frame)
    label_id = svm_predict_with_unknown(model, features)
    label = classes[label_id]

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Material Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()