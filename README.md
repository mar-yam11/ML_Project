📦 Material Stream Identification (Simple Student Version)

Very simple pipeline: clean → augment → train → evaluate → realtime.

Quick Start

1) Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Prepare Data

```bash
# Put class folders under data/raw/ (glass, paper, cardboard, plastic, metal, trash)

# Clean corrupt images → data/cleaned/
python -m src.clean

# Augment & balance (≈600 per class) → data/augmented/
python -m src.augmentation
```

3) Train Models (SVM + k-NN)

```bash
# Combined HOG + Color features
python -m src.train --data-dir data/augmented --feature-type hog_color
```

This saves artifacts in models/: scaler.joblib, svm_model.joblib, knn_model.joblib, class_map.json

4) Evaluate (Notebook)

- Open notebooks/evaluation.ipynb and run all cells.
- It shows accuracy, confusion matrices, and Unknown rejection.

5) Realtime App (Webcam)

```bash
# Use SVM (default)
python -m src.app_realtime --models-dir models --threshold 0.6

# Or use k-NN
python -m src.app_realtime --models-dir models --threshold 0.6 --use-knn
```

- If confidence < threshold → label is "Unknown".

6) Make Submission Zip (simple)

```bash
python scripts/make_submission.py --ta Ahmed --group G5 --ids 202001 202002 202003
```

Files Included

- src/: code (clean, augmentation, features, train, predict, realtime app)
- data/: raw, cleaned, augmented (created by scripts)
- models/: saved scaler and models (after training)
- notebooks/evaluation.ipynb: quick comparison and rejection
- docs/REPORT.md: fill and export to PDF
- scripts/make_submission.py: packs zip

Notes

- Keep it simple. You can tune: `--feature-type`, `--svm-c`, `--svm-kernel`, `--knn-k`, and rejection threshold.
- Unknown class is handled by thresholding max probability. Map it to class 6 in your final app/report.
