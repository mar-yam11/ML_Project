# Data Pipeline (Simple)

This is a quick, student-friendly guide.

## Folders

- raw/: original images in class folders (glass, paper, cardboard, plastic, metal, trash)
- cleaned/: auto-generated JPEGs after validation
- augmented/: balanced set after augmentation

## Steps

1. Clean raw images (makes cleaned/):

```bash
python -m src.clean
```

2. Augment to target count per class (e.g., 600):

```bash
python -m src.augmentation
```

3. Train models using augmented data:

```bash
python -m src.train --data-dir data/augmented --feature-type hog_color
```

Notes:

- Unknown class is handled at inference with a probability threshold (see app and notebook).
- You can change target count in `src/augmentation.py` (`augment_data(..., class_count=600)`).
