import numpy as np

def svm_predict(model, features):
    probs = model.predict_proba([features])[0]
    max_prob = max(probs)

    if max_prob < 0.6:
        return 6  # Unknown

    return np.argmax(probs)