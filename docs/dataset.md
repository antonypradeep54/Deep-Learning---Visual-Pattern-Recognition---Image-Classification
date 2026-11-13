# Dataset preparation — Package 02.2A

## Source and licensing

Candidate source: Kaggle **Dogs vs. Cats** image-classification competition dataset. Confirm the exact download URL, access requirements, terms, and permitted redistribution before downloading. The dataset is **not** bundled in this repository. Do not commit downloaded images, CSV manifests containing local-sensitive paths, or archives.

## Expected layout

The pipeline supports either class subdirectories (`Cat/`, `Dog/`, or plural variants) or filenames beginning with `cat.` / `dog.` (also `cat_` / `dog_`). Example:

```text
data/raw/pets/
  cat.0.jpg
  dog.0.jpg
```

## Setup and run

```bash
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py --input data/raw/pets --output data/processed --seed 42
python -m pytest -q
```

The command generates `data/processed/splits.csv` with relative path, label, SHA-256 checksum and split; plus `data/processed/rejected.csv` with reasons for excluded images. Both are ignored by Git.

## Quality and leakage controls

- Validate that each candidate image can be decoded and converted to RGB.
- Exclude files without recognizable Cat/Dog labels.
- Use SHA-256 to identify **exact file duplicates** before splitting; reject both images when identical bytes have conflicting labels.
- Sort filenames and shuffle each class with seed 42, then split each class approximately **70% train / 15% validation / 15% test**.
- Keep the test partition untouched during architecture selection and tuning.
- Inspect perceptual near-duplicates manually or in a subsequent enhancement; SHA-256 does **not** catch resized or re-encoded duplicates.

## Scope boundaries

This package prepares manifests only. Image resizing to 128 × 128, normalization, training-only augmentation, batched `tf.data` loading, model training and independent evaluation belong to subsequent packages. A fixed seed helps repeat the split on the same files; the manifest and source dataset version should be retained for full reproducibility.
