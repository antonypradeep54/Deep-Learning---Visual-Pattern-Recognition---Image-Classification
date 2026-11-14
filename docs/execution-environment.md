# Phase 1 — Execution environment

Reported from a successful Google Colab environment on 2026-11-14:

| Component | Version |
| --- | --- |
| Python | 3.13.16 |
| TensorFlow | 2.21.0 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.0 |
| Pillow | 11.3.0 |
| pytest | 8.4.2 |

The published notebook was structurally and syntax checked but has not been executed end-to-end after publication cleanup. The reported model results come from the original completed Colab run. Installation from `requirements.txt` on a fresh environment has not yet been independently verified.

The dataset archive, saved CNN checkpoint, and baseline evaluation artifacts are stored outside this repository in the author's Google Drive. Running the entire notebook retrains the model and may overwrite the existing checkpoint. The notebook assumes a Colab environment and can use `CNN_PROJECT_DIR` to override the default Google Drive project directory.
