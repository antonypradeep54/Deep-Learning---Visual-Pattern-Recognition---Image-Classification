# Phase 1 — Custom CNN: Model Evaluation

**Project:** Cats vs Dogs — Visual Pattern Recognition and Image Classification  
**Status:** Completed baseline experiment; documentation prepared for review  
**Evaluation source:** Completed Google Colab run, 14 November 2026. The cleaned publication notebook has not been independently rerun end-to-end.

## 1. Objective and scope

Build and evaluate a convolutional neural network (CNN) that classifies an image as **Cat** or **Dog**. This is a two-class demonstration, not a general-purpose animal recognizer. The model outputs a sigmoid score interpreted as the estimated probability of **Dog**. The default decision threshold is **0.50**.

## 2. Dataset and quality checks

- Source: Kaggle Dogs vs Cats training archive (`train.zip`), held outside the repository.
- Images examined: **25,000**.
- Exact duplicates rejected: **6**; images accepted: **24,994**.
- Accepted class counts: **12,497 Cat**, **12,497 Dog**.
- Dataset split: deterministic, stratified by class, using seed **42**.

| Split | Cat | Dog | Total |
| --- | ---: | ---: | ---: |
| Train | 8,747 | 8,747 | **17,494** |
| Validation | 1,874 | 1,874 | **3,748** |
| Independent test | 1,876 | 1,876 | **3,752** |
| **Total** | **12,497** | **12,497** | **24,994** |

Image decoding and exact-byte duplicate checks are implemented in `src/data_pipeline.py`. SHA-256 does **not** detect visually similar but re-encoded or resized images. The independent test partition was held out from model training and callback-based selection.

## 3. Architecture and training

- **Input:** 128 × 128 × 3 RGB images; pixel values scaled to [0, 1].
- **Training-only augmentation:** horizontal flip, rotation factor 0.1, zoom factor 0.1.
- **Feature extraction:** three convolutional blocks with 32, 64, and 128 filters, each followed by 2 × 2 max pooling.
- **Classification:** Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid).
- **Trainable parameters:** approximately **3.30 million** (3,304,769 in the recorded model summary).
- **Optimizer / loss:** Adam (learning rate 0.001) / binary cross-entropy.
- **Batch size:** 32; **maximum epochs:** 15; **early stopping patience:** 3 epochs, monitoring validation loss.
- **Checkpoint selection:** lowest validation loss, saved as `best_cnn_model.keras` in the author's Google Drive.

The model was trained from scratch, without a pretrained image backbone. The checkpoint was verified to exist (reported size: **37.88 MB**). The checkpoint and image dataset are not included in GitHub.

## 4. Independent test results

The best saved checkpoint was reloaded and evaluated on the **3,752-image** held-out test partition. Dog is the positive class; class predictions use a threshold of **0.50**.

| Metric | Result |
| --- | ---: |
| Accuracy | **0.8761 (87.61%)** |
| Precision (Dog) | **0.8904 (89.04%)** |
| Recall (Dog) | **0.8577 (85.77%)** |
| F1-score (Dog) | **0.8737 (87.37%)** |
| ROC-AUC | **0.9477** |

**Interpretation:** The model correctly classifies approximately 88 out of 100 images from this held-out dataset. ROC-AUC summarizes its ability to rank Dog examples above Cat examples across classification thresholds; it is not the same as classification accuracy.

## 5. Confusion matrix and error analysis

| Actual / Predicted | Cat | Dog | Total |
| --- | ---: | ---: | ---: |
| **Cat** | 1,678 | 198 | 1,876 |
| **Dog** | 267 | 1,609 | 1,876 |
| **Total** | 1,945 | 1,807 | **3,752** |

- **True negatives (Cat correctly identified):** 1,678.
- **False positives (Cat classified as Dog):** 198.
- **False negatives (Dog classified as Cat):** 267.
- **True positives (Dog correctly identified):** 1,609.
- **Total correct:** 3,287; **total incorrect:** 465 (**12.39%**).

The model misses more actual Dogs than it falsely labels Cats as Dogs. In a product where missing a Dog matters more than a false Dog alert, a different threshold could be explored using **validation data**, with the associated precision–recall trade-off documented.

The notebook displays three examples of each error category. These visual examples are exploratory; without a systematic annotation study, we should **not** assert that lighting, occlusion, background, or pose caused the errors.

## 6. Interactive prediction demonstration

The final notebook cell allows a Colab user to upload an image, reload the saved CNN checkpoint, apply the same image resizing and normalization used for training, and display a Cat/Dog prediction and score. The score is a model output, **not a calibrated guarantee**. Images outside the two training classes will still receive a Cat or Dog prediction.

## 7. Limitations and reproducibility

1. **Generalization:** Performance was measured on the held-out partition of the same source dataset, not on a separate external or production dataset.
2. **Dataset leakage:** Exact duplicate detection is implemented, but perceptual near-duplicates were not comprehensively audited.
3. **Training variability:** Stochastic augmentation and hardware/runtime effects can produce different results on retraining, even with a fixed split seed.
4. **Probability calibration:** Calibration of sigmoid outputs has not been assessed.
5. **Performance efficiency:** Inference latency, memory consumption, and deployment performance have not been benchmarked.
6. **Reproducibility:** The original Colab experiment completed successfully. The cleaned publication notebook and fresh-environment dependency installation have **not** been verified end-to-end. The source images, manifest copies, and trained checkpoint are kept in Google Drive rather than distributed in GitHub.
7. **Test-set discipline:** Future architecture and threshold comparisons should be decided using validation results; avoid repeated test-driven tuning.

## 8. Preserved experiment artifacts

The successful run saved these files under the author's Drive project directory at `results/baseline_cnn/`:

- `training_history.csv`
- `test_metrics.json`
- `confusion_matrix.csv`
- `model_architecture.json`
- `experiment_config.json`

The trained model is stored separately at `checkpoints/best_cnn_model.keras`. These are **reported saved artifacts**, not files bundled in this repository.

## 9. Next phase (not part of this evaluation)

**Phase 2 — Transfer Learning:** Consider MobileNetV2 with pretrained ImageNet weights, compare validation performance, training cost, model size, and potential inference efficiency against this baseline. Do not present Phase 2 outcomes until those experiments are executed and verified.