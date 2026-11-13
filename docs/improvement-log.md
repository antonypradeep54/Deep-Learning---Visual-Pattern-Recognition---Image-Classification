# CNN modernization — improvement log

This document tracks development findings and their resolution. Entries below are **planned**, not completed or experimentally verified.

| ID | Finding / work item | Planned action | Package | Status |
| --- | --- | --- | --- | --- |
| CNN-01 | Historical dataset files and images are not included | Identify permitted dataset source and document download and preparation | 02 | Open |
| CNN-02 | Historical training uses `fit_generator()` | Update modern implementation to a supported TensorFlow/Keras training API | 02 | Open |
| CNN-03 | The file named `test_dog_cat.csv` is used for validation during training | Establish distinct training, validation, and held-out test splits | 02 | Open |
| CNN-04 | Reproducibility settings and dependencies are undocumented | Pin tested dependencies, configure seeds, and document run instructions | 02 | Open |
| CNN-05 | Model performance has not been independently verified | Run evaluation and report accuracy, precision, recall, F1 and confusion matrix | 03 | Open |
| CNN-06 | Model generalization and error patterns are not characterized | Review failure cases, class balance, and possible improvements | 03 | Open |
| CNN-07 | Original HTML export is not in the repository | Review contents and redistribution rights before considering upload | 01 | Pending review |

## Change principles

- Preserve `original/pet_classification_original.py` as the historical baseline.
- Implement updates in separate `src/` and `notebooks/` files when ready.
- Do not report model metrics until a reproducible evaluation has been run.
- Update this log as each issue is resolved.
