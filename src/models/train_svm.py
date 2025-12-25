# Import NumPy to load feature arrays
import numpy as np
# Import joblib to save trained models
import joblib
# Import SVM classifier
from sklearn.svm import SVC
# Import train/validation split utility
from sklearn.model_selection import train_test_split
# Import standardization transformer
from sklearn.preprocessing import StandardScaler
# Import Pipeline to chain preprocessing and model
from sklearn.pipeline import Pipeline
# Import accuracy metric
from sklearn.metrics import accuracy_score

 # Load precomputed feature matrix X and labels y
X = np.load("src/models/X.npy")
 y = np.load("src/models/y.npy")

 # Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

 # Build a pipeline: scale features then apply linear SVM
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True))
])

 # Train the SVM model on the training set
model.fit(X_train, y_train)

 # Evaluate on the validation set
predictions = model.predict(X_val)
acc = accuracy_score(y_val, predictions)

 # Print validation accuracy
print("SVM Accuracy:", acc)

 # Save the trained pipeline to disk
joblib.dump(model, "src/models/saved_models/svm_model.joblib")
