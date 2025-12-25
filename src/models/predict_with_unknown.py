# Import NumPy for working with arrays
import numpy as np

 # Predict class index using SVM and return 'Unknown' if confidence is low
def svm_predict(model, features):
    # Get probability distribution over classes for the input features
    probs = model.predict_proba([features])[0]
    # Find the highest predicted probability
    max_prob = max(probs)

    # If the model is not confident enough, mark as Unknown (index 6)
    if max_prob < 0.6:
        return 6  # Unknown

    # Otherwise return the index of the class with highest probability
    return np.argmax(probs)