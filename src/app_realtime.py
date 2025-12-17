import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from .features import hog_features, color_hist_features
from .predict import predict_proba


def load(models_dir: Path):
    scaler = joblib.load(models_dir / 'scaler.joblib')
    with open(models_dir / 'class_map.json', 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    with open(models_dir / 'feature_config.json', 'r', encoding='utf-8') as f:
        feat_conf = json.load(f)
    feature_type = feat_conf.get('feature_type', 'hog_color')
    # Default to SVM for realtime; provide option via arg
    model = joblib.load(models_dir / 'svm_model.joblib')
    return scaler, model, class_map, feature_type


def frame_features(frame_bgr: np.ndarray, feature_type: str) -> np.ndarray:
    if feature_type == 'hog':
        f = hog_features(frame_bgr)
    elif feature_type == 'color':
        f = color_hist_features(frame_bgr)
    else:
        f = np.concatenate([hog_features(frame_bgr), color_hist_features(frame_bgr)]).astype(np.float32)
    return f


def main():
    ap = argparse.ArgumentParser(description='Realtime MSI classification')
    ap.add_argument('--models-dir', type=str, default='models')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--use-knn', action='store_true', help='Use k-NN instead of SVM')
    args = ap.parse_args()

    scaler, model, class_map, feature_type = load(Path(args.models_dir))
    if args.use_knn:
        model = joblib.load(Path(args.models_dir) / 'knn_model.joblib')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    classes = [class_map[str(i)] if isinstance(class_map, dict) and str(i) in class_map else class_map[i] for i in range(len(class_map))]

    print('Press Q to quit.')
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat = frame_features(frame, feature_type)
        Xs = scaler.transform(feat.reshape(1, -1))
        try:
            proba = predict_proba(model, Xs)
            conf = float(proba.max(axis=1)[0])
            pred = int(proba.argmax(axis=1)[0])
        except Exception:
            # For models without predict_proba (should not happen with current choices)
            pred = int(model.predict(Xs)[0])
            conf = 1.0

        label = 'Unknown'
        color = (0, 0, 255)
        if conf >= args.threshold:
            label = classes[pred]
            color = (0, 200, 0)

        cv2.putText(frame, f'{label} ({conf:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.imshow('MSI Realtime', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
