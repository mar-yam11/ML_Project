# Setup Instructions - Creating Models Folder

This document explains how to initialize the ML models and generate the required files in the `models/` folder.

## Prerequisites

- Python 3.x installed
- All dependencies from `requirements.txt` installed

```bash
pip install -r requirements.txt
```

## Step 1: Build Datasets

First, create the feature datasets from the augmented images:

```bash
python -m src.models.build_datasets
```

This will generate:

- `models/X.npy` - Feature vectors
- `models/y.npy` - Labels

## Step 2: Train Models

Create a folder saved_models inside src/models/,
Then train the SVM and KNN models:

```bash
python -m src.models.train_svm
python -m src.models.train_knn
```

This will generate:

- `models/svm_model.joblib` - Trained SVM model
- `models/knn_model.joblib` - Trained KNN model

## Output

After completing all steps, your `models/` folder will contain all trained models and necessary configuration files.

## Notes

- Run from the project root directory
- Use `-m` flag to run as modules (ensures correct imports)
- Make sure `data/augmented/` folder has images before running
