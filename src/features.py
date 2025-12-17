import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from skimage.feature import hog


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def list_images(folder: Path) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if Path(name).suffix.lower() in IMAGE_EXTS:
            files.append(name)
    return files


def _read_image_bgr(img_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return img


def color_hist_features(img_bgr: np.ndarray, hsv_bins: Tuple[int, int, int] = (16, 8, 8)) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = hsv_bins
    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    hist = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist


def hog_features(img_bgr: np.ndarray, image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, image_size)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_image_features(
    img_path: Path,
    feature_type: str = 'hog_color',
    image_size: Tuple[int, int] = (128, 128),
    hsv_bins: Tuple[int, int, int] = (16, 8, 8),
) -> Optional[np.ndarray]:
    img = _read_image_bgr(img_path)
    if img is None:
        return None
    if feature_type == 'hog':
        return hog_features(img, image_size=image_size)
    elif feature_type == 'color':
        return color_hist_features(img, hsv_bins=hsv_bins)
    elif feature_type in ('hog_color', 'color_hog'):
        h = hog_features(img, image_size=image_size)
        c = color_hist_features(img, hsv_bins=hsv_bins)
        return np.concatenate([h, c]).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def extract_dataset_features(
    data_dir: Path,
    feature_type: str = 'hog_color',
    image_size: Tuple[int, int] = (128, 128),
    hsv_bins: Tuple[int, int, int] = (16, 8, 8),
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    folders: List[Path] = [p for p in Path(data_dir).iterdir() if p.is_dir()]
    folders.sort(key=lambda p: p.name)

    X: List[np.ndarray] = []
    y: List[int] = []
    class_map: Dict[int, str] = {}

    for idx, folder in enumerate(folders):
        class_map[idx] = folder.name
        for img_name in list_images(folder):
            feat = extract_image_features(folder / img_name, feature_type=feature_type, image_size=image_size, hsv_bins=hsv_bins)
            if feat is not None:
                X.append(feat)
                y.append(idx)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr, class_map
