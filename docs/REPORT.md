# Final Report (Template)

Fill in the sections briefly and clearly. Keep it simple.

## 1. Introduction

- Goal: Automated Material Stream Identification (MSI) using classic ML.
- Classes: glass, paper, cardboard, plastic, metal, trash, unknown.

## 2. Dataset and Preprocessing

- Source and structure (class folders).
- Cleaning: removed corrupt images, converted to JPEG.
- Train/Val split: stratified split (e.g., 80/20).

## 3. Data Augmentation

- Techniques: rotation, flips, color jitter, resize to 64–128.
- Per-class target count after augmentation (e.g., ~600).
- Rationale: increase robustness to orientation/light.

## 4. Feature Extraction

- Methods: HOG (shape/edges) and HSV color histograms.
- Final descriptor: HOG + Color concat. Provide dimensions.
- Normalization: StandardScaler fitted on train.

## 5. Models

- SVM: kernel = RBF, C = 2.0 (tuned lightly).
- k-NN: k = 7, weights = distance.
- Training details: brief notes (defaults otherwise).

## 6. Unknown-class Handling

- Probability threshold τ (e.g., 0.6).
- Predict Unknown when max probability < τ.
- Trade-off: higher τ → fewer accepted predictions, higher accepted accuracy.

## 7. Evaluation

- Metrics: accuracy, confusion matrices per model.
- Key results: SVM vs k-NN accuracy on validation split.
- Rejection analysis: coverage vs accepted accuracy at τ.

## 8. Deployment

- Realtime app: OpenCV webcam, HOG+Color per-frame, threshold rejection.
- Performance notes: FPS, any constraints.

## 9. Discussion

- What worked: features that helped.
- What didn’t: failure cases (e.g., glare, blur).
- Future improvements: better descriptors, calibration, more data.

## 10. Conclusion

- Short summary and main takeaways.

## Appendix

- Training commands and versions.
- Hardware used.
