import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import extract_dataset_features


def train_and_evaluate(
    data_dir: Path,
    out_dir: Path,
    feature_type: str = 'hog_color',
    test_size: float = 0.2,
    random_state: int = 42,
    svm_c: float = 2.0,
    svm_kernel: str = 'rbf',
    knn_k: int = 7,
    knn_weights: str = 'distance',
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        fallback = Path('data/cleaned')
        if fallback.exists():
            print(f"Warning: {data_dir} not found. Using {fallback} instead.")
            data_dir = fallback
        else:
            raise RuntimeError(f"Data not found at {data_dir} or {fallback}. Run clean.py and augmentation.py first.")
    X, y, class_map = extract_dataset_features(data_dir, feature_type=feature_type)
    if len(X) == 0:
        raise RuntimeError(f"No images/features found under {data_dir}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    svm = SVC(C=svm_c, kernel=svm_kernel, gamma='scale', probability=True, class_weight='balanced')
    svm.fit(X_train_s, y_train)
    svm_pred = svm.predict(X_val_s)
    svm_acc = accuracy_score(y_val, svm_pred)

    knn = KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)
    knn.fit(X_train_s, y_train)
    knn_pred = knn.predict(X_val_s)
    knn_acc = accuracy_score(y_val, knn_pred)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / 'scaler.joblib')
    joblib.dump(svm, out_dir / 'svm_model.joblib')
    joblib.dump(knn, out_dir / 'knn_model.joblib')
    with open(out_dir / 'class_map.json', 'w', encoding='utf-8') as f:
        json.dump(class_map, f, indent=2)
    with open(out_dir / 'feature_config.json', 'w', encoding='utf-8') as f:
        json.dump({'feature_type': feature_type}, f, indent=2)

    print("\n=== Validation Results ===")
    print(f"SVM accuracy: {svm_acc:.4f}")
    print(f"k-NN accuracy: {knn_acc:.4f}")
    print("\n-- SVM report --")
    print(classification_report(y_val, svm_pred, digits=4))
    print("\n-- k-NN report --")
    print(classification_report(y_val, knn_pred, digits=4))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train SVM and k-NN on extracted features')
    p.add_argument('--data-dir', type=str, default='data/augmented', help='Dataset root dir (class subfolders)')
    p.add_argument('--out-dir', type=str, default='models', help='Output dir for models')
    p.add_argument('--feature-type', type=str, default='hog_color', choices=['hog', 'color', 'hog_color'], help='Feature descriptor')
    p.add_argument('--test-size', type=float, default=0.2, help='Validation split size')
    p.add_argument('--random-state', type=int, default=42, help='Random seed')
    p.add_argument('--svm-c', type=float, default=2.0, help='SVM C')
    p.add_argument('--svm-kernel', type=str, default='rbf', choices=['rbf', 'linear', 'poly', 'sigmoid'], help='SVM kernel')
    p.add_argument('--knn-k', type=int, default=7, help='kNN k')
    p.add_argument('--knn-weights', type=str, default='distance', choices=['uniform', 'distance'], help='kNN weighting')
    return p.parse_args()


def main():
    args = parse_args()
    train_and_evaluate(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        feature_type=args.feature_type,
        test_size=args.test_size,
        random_state=args.random_state,
        svm_c=args.svm_c,
        svm_kernel=args.svm_kernel,
        knn_k=args.knn_k,
        knn_weights=args.knn_weights,
    )


if __name__ == '__main__':
    main()
