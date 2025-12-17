import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from .features import extract_image_features


def load_artifacts(models_dir: Path):
    scaler = joblib.load(models_dir / 'scaler.joblib')
    svm = joblib.load(models_dir / 'svm_model.joblib')
    knn = joblib.load(models_dir / 'knn_model.joblib')
    with open(models_dir / 'class_map.json', 'r', encoding='utf-8') as f:
        class_map: Dict[int, str] = json.load(f)
    with open(models_dir / 'feature_config.json', 'r', encoding='utf-8') as f:
        feat_conf = json.load(f)
    return scaler, svm, knn, class_map, feat_conf


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    raise ValueError('Model does not support predict_proba')


def apply_rejection(probas: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    max_proba = probas.max(axis=1)
    labels = probas.argmax(axis=1)
    reject_mask = max_proba < threshold
    labels_rej = labels.copy()
    labels_rej[reject_mask] = -1  # -1 denotes Unknown
    return labels_rej, max_proba


def predict_image_with_reject(
    img_path: Path,
    scaler,
    model,
    feature_type: str = 'hog_color',
    threshold: float = 0.6,
):
    feat = extract_image_features(img_path, feature_type=feature_type)
    if feat is None:
        return -1, 0.0
    Xs = scaler.transform(feat.reshape(1, -1))
    probas = predict_proba(model, Xs)
    y_hat, conf = apply_rejection(probas, threshold)
    return int(y_hat[0]), float(conf[0])
