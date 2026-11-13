# Deep Learning — Visual Pattern Recognition & Image Classification

Exploring visual pattern recognition and image classification using deep learning and convolutional neural networks (CNNs), with **Cats vs Dogs** classification as a practical case study.

## Project objective

Develop a model that learns visual patterns from labeled images and predicts whether a new image contains a cat or a dog. This repository preserves an earlier CNN learning project and documents its planned modernization and evaluation.

## Original CNN approach

The historical implementation uses TensorFlow/Keras with:
- **Input:** 60 × 60 RGB images
- **Normalization:** Batch normalization at the model input
- **Feature extraction:** Two convolutional layers (32 and 64 filters, 5 × 5 kernels), each followed by 2 × 2 max pooling
- **Classification:** Flatten → Dense (32, ReLU) → Dropout (0.4) → Dense (2, softmax)
- **Training configuration:** Adam optimizer, categorical cross-entropy loss, accuracy metric
- **Training stages in source code:** 25 epochs using the initial training dataset, followed by 50 epochs using a larger Kaggle training dataset

These details describe the historical **source code**, not newly verified execution results.

## Repository structure

| Location | Purpose |
| --- | --- |
| `original/` | Preserved historical implementation and notes |
| `docs/improvement-log.md` | Development findings, planned work, and resolution tracking |
| `notebooks/` | Planned reproducible experiments and interactive analysis |
| `src/` | Planned modernized Python implementation |

The `notebooks/` and `src/` directories will be created when their first files are ready.

## Development roadmap

1. **Package 01 — Historical baseline:** Preserve original code and document its architecture.
2. **Package 02 — Modernization:** Establish reproducible data preparation, compatible training code, and train/validation/test separation.
3. **Package 03 — Evaluation and communication:** Run independent evaluation, document verified results and insights, and prepare the recruiter-facing project narrative.

## Project status

**In development — historical baseline documentation.** Modernized training and independent evaluation have not yet been completed. Verified results will be added after execution.

## License

See [LICENSE](LICENSE). Dataset and third-party material rights are separate from the repository's code license.
