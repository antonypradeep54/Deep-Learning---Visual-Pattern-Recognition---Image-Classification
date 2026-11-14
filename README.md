# Deep Learning — Visual Pattern Recognition & Image Classification

**Cats vs Dogs | Custom Convolutional Neural Network (CNN) | TensorFlow / Keras**

## Project overview

Can a neural network learn visual features directly from images and distinguish between two categories? This project explores that question through **Cat vs Dog image classification**, using a custom CNN trained from scratch and evaluated on a held-out test set.

The project demonstrates an end-to-end image-classification workflow: image quality checks, duplicate detection, stratified data splitting, preprocessing, model training, independent evaluation, error analysis, and interactive inference in Google Colab.

**Status:** Phase 1 (custom CNN baseline) completed and evaluated. Phase 2 (transfer learning) is a proposed future experiment, not an implemented result.

## Why a CNN?

Unlike conventional machine-learning approaches that often rely on manually designed image features, CNNs learn spatial patterns such as edges, textures, and increasingly complex shapes directly from image pixels. Convolution and pooling also help the model recognize local visual patterns across an image. This makes a CNN a relevant baseline for image classification; it does **not** establish that a CNN will outperform every classical ML approach without a comparative experiment.

## Results — independent held-out test set

| Metric | Result |
| --- | ---: |
| Accuracy | **87.61%** |
| Precision (Dog) | **89.04%** |
| Recall (Dog) | **85.77%** |
| F1-score (Dog) | **87.37%** |
| ROC-AUC | **0.9477** |
| Test images | **3,752** |

The model correctly classified **3,287** of **3,752** test images. With Dog as the positive class, it classified **198 Cats as Dogs** and **267 Dogs as Cats**. These results were measured in the original completed Google Colab experiment; the cleaned publication notebook has not been independently rerun end-to-end.

For the confusion matrix, metric definitions, interpretation, and caveats, see [Model evaluation](docs/model-evaluation.md).

## Dataset and preprocessing

- **Source:** Kaggle Dogs vs Cats training archive; images are **not** redistributed in this repository.
- **Examined:** 25,000 images; **6 exact duplicates** rejected; **24,994** accepted.
- **Class balance:** 12,497 Cats and 12,497 Dogs.
- **Splits:** 17,494 training / 3,748 validation / 3,752 independent test images, using a deterministic stratified split with seed 42.
- **Input:** 128 × 128 RGB, normalized to [0, 1].
- **Training-only augmentation:** horizontal flip, rotation, and zoom.

Exact-byte duplicate detection does not eliminate visually similar images that have been resized or re-encoded. See [Dataset documentation](docs/dataset.md).

## Model design

A custom CNN was trained **from scratch**, without a pretrained backbone:

`128×128 RGB → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Sigmoid`

- Approximately **3.30 million trainable parameters**.
- **Adam** optimizer (learning rate 0.001) and **binary cross-entropy** loss.
- Batch size **32**, up to **15 epochs**, early stopping patience **3**, and best-checkpoint selection by validation loss.
- The sigmoid output is interpreted as a Dog score; classification uses a **0.50 threshold**. This score has **not** been assessed for probability calibration.

## Try the notebook

1. Open the [Phase 1 Colab notebook](notebooks/Deep-Learning--Visual-Pattern-Recognition--Image-Classification.ipynb) in Google Colab.
2. Consult the [execution environment](docs/execution-environment.md) and [`requirements.txt`](requirements.txt).
3. Supply your own authorized copy of the Kaggle dataset and configure the project folder in Google Drive; the notebook uses `CNN_PROJECT_DIR` to override its default project location.
4. Run the relevant preparation, training, and evaluation cells if reproducing the experiment. **Running training again can overwrite the existing checkpoint and produce different metrics.**
5. For interactive classification, the notebook includes an image-upload cell that loads the saved checkpoint and predicts Cat or Dog. It requires access to the checkpoint, which is **not included in this repository**.

The original completed Colab run produced the reported metrics. A fresh, end-to-end run of the cleaned notebook and a clean-environment dependency installation have **not** yet been independently verified. Do not interpret the notebook as a one-click demo without supplying the dataset and checkpoint.

## Product perspective

Visual classification can underpin product capabilities such as image categorization, content organization, and human-in-the-loop review. This two-class project is a **learning and evaluation case study**, not a production-ready general animal classifier.

The product lesson is that **accuracy alone is insufficient**. Precision and recall expose different types of mistakes, while the confusion matrix helps assess whether the model's failure modes are acceptable for a specific use case. Any threshold adjustment should be selected using validation data and a defined product objective, not by repeatedly optimizing against the held-out test set.

## Limitations and future work

- Evaluation uses a held-out portion of the **same source dataset**, not a separate real-world image collection.
- Visually similar near-duplicates were not comprehensively audited.
- Inference latency, deployment costs, and confidence calibration were not benchmarked.
- The classifier always outputs Cat or Dog, even for unrelated images.
- **Future Phase 2:** evaluate a pretrained **MobileNetV2** transfer-learning model against this custom CNN on validation performance, model size, training cost, and inference efficiency. No Phase 2 metrics are claimed.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`notebooks/Deep-Learning--Visual-Pattern-Recognition--Image-Classification.ipynb`](notebooks/Deep-Learning--Visual-Pattern-Recognition--Image-Classification.ipynb) | Phase 1 Colab notebook and interactive prediction |
| [`docs/model-evaluation.md`](docs/model-evaluation.md) | Verified metrics, confusion matrix, and limitations |
| [`docs/dataset.md`](docs/dataset.md) | Dataset preparation, leakage controls, and split design |
| [`docs/execution-environment.md`](docs/execution-environment.md) | Recorded Colab dependency versions and reproduction caveats |
| [`src/data_pipeline.py`](src/data_pipeline.py) | Image validation, duplicate checks, and split manifest logic |
| [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py) | Dataset manifest preparation command |
| [`original/`](original/) | Preserved historical implementation |

## License and data rights

See [LICENSE](LICENSE). The Kaggle dataset, trained checkpoint, and saved Google Drive experiment artifacts are not distributed in this repository; their rights and access requirements are separate from the repository's code license.