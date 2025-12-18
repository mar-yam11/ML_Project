# Team Responsibilities - Code Explanation

## Part 1: Data Pipeline (Student 1)

**Student 1 explains:**

- `src/clean.py` - data cleaning pipeline, image validation, corrupt file detection
- `src/augmentation.py` - data augmentation techniques (rotation, flip, color jitter), class balancing
- `data/README.md` - data preparation workflow and folder structure

**Key concepts to explain:**

- How cleaning removes corrupt images and standardizes formats
- Augmentation techniques and their purpose (robustness to variations)
- Class balancing strategy to reach target count (e.g., 600 per class)
- Output structure: cleaned/ and augmented/ folders

---

## Part 2: Feature Extraction & Model Training (Student 2)

**Student 2 explains:**

- `src/features.py` - feature extraction module (HOG, color histograms, combined)
- `src/train.py` - SVM and k-NN training pipeline, hyperparameter tuning, validation
- `src/predict.py` - prediction utilities, probability-based Unknown-class rejection mechanism
- `notebooks/evaluation.ipynb` - model evaluation, accuracy comparison, confusion matrices, rejection analysis

**Key concepts to explain:**

- Feature descriptors: HOG (shape/edges) vs Color histograms (appearance) vs Combined
- SVM kernel choice, C parameter tuning
- k-NN k selection and distance weighting
- StandardScaler normalization and why it's needed
- Unknown-class rejection: probability threshold mechanism
- Evaluation metrics: accuracy, confusion matrix, classification report
- Trade-off between coverage and accepted accuracy with rejection threshold

---

## Part 3: Real-Time Application & Deployment (Student 3)

**Student 3 explains:**

- `src/app_realtime.py` - real-time webcam classification application
- `scripts/make_submission.py` - submission packaging script
- `docs/REPORT.md` - report template structure
- `README.md` - project quickstart and usage instructions

**Key concepts to explain:**

- Real-time frame processing pipeline
- Per-frame feature extraction (same as training)
- Model loading and inference
- Probability threshold-based Unknown classification in live video
- Deployment considerations: FPS, stability, edge cases
- System integration and end-to-end workflow

---

## Cross-Cutting Responsibilities

**All students must understand:**

- `src/__init__.py` and `requirements.txt` - Python packaging and dependencies
- How data flows through the entire pipeline
- Unknown-class (ID 6) handling across all stages
- Module imports and how to run scripts

---

## Presentation Order Suggestion

1. **P1 (Student 1)**: 2–3 minutes on data cleaning, augmentation, class balancing
2. **P2 (Student 2)**: 4–5 minutes on features, training, evaluation, Unknown rejection
3. **P3 (Student 3)**: 2–3 minutes on deployment, real-time app, packaging

**Total: ~10 minutes**
