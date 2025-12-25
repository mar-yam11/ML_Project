# Import NumPy to load feature arrays
import numpy as np
# Import joblib to save trained models
import joblib
# Import kNN classifier from scikit-learn
from sklearn.neighbors import KNeighborsClassifier
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

 # Build a pipeline: scale features then apply kNN classifier
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
      n_neighbors=3,
      weights="distance"
    ))
])

 # Train the kNN model on the training set
model.fit(X_train, y_train)

 # Evaluate on the validation set
predictions = model.predict(X_val)
acc = accuracy_score(y_val, predictions)

 # Print validation accuracy
print("kNN Accuracy:", acc)

 # Save the trained pipeline to disk
joblib.dump(model, "src/models/saved_models/knn_model.joblib")
