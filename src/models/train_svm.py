import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X = np.load("src/models/X.npy")
y = np.load("src/models/y.npy")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True))
])

model.fit(X_train, y_train)

predictions = model.predict(X_val)
acc = accuracy_score(y_val, predictions)

print("SVM Accuracy:", acc)

joblib.dump(model, "src/models/saved_models/svm_model.joblib")
