"""Deterministic Cats vs Dogs dataset inventory and split manifests."""
from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

CLASSES = ("Cat", "Dog")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _label(path: Path) -> str | None:
    """Infer label from Cat/Dog parent folder or cat./dog. filename."""
    parent = path.parent.name.lower()
    if parent in {"cat", "cats"}:
        return "Cat"
    if parent in {"dog", "dogs"}:
        return "Dog"
    stem = path.stem.lower()
    for prefix, label in (("cat.", "Cat"), ("dog.", "Dog"), ("cat_", "Cat"), ("dog_", "Dog")):
        if stem.startswith(prefix):
            return label
    return None


def inventory(root: Path) -> tuple[list[dict], list[dict]]:
    """Validate image decoding, infer labels and identify exact content duplicates."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    rows, rejected = [], []
    seen_hashes: dict[str, dict] = {}
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    for path in paths:
        rel = path.relative_to(root).as_posix()
        label = _label(path)
        if label is None:
            rejected.append({"path": rel, "reason": "unrecognized_label"})
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB").load()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            rejected.append({"path": rel, "reason": f"invalid_image:{type(exc).__name__}"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            prior = seen_hashes[digest]
            reason = "duplicate_conflicting_label" if prior["label"] != label else "duplicate_exact"
            rejected.append({"path": rel, "reason": reason})
            # Conflicting labels invalidate both records rather than silently choosing one.
            if reason == "duplicate_conflicting_label":
                if prior in rows:
                    rows.remove(prior)
                rejected.append({"path": prior["path"], "reason": "duplicate_conflicting_label"})
                seen_hashes[digest] = {"path": prior["path"], "label": "CONFLICT"}
            continue
        row = {"path": rel, "label": label, "sha256": digest}
        rows.append(row)
        seen_hashes[digest] = row
    return rows, rejected


def stratified_split(rows: list[dict], seed: int = 42) -> list[dict]:
    """Split each class into 70/15/15 with deterministic ordering and seed."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    if set(grouped) != set(CLASSES):
        raise ValueError("Both Cat and Dog classes are required")
    result = []
    for label in CLASSES:
        items = sorted(grouped[label], key=lambda r: r["path"])
        if len(items) < 7:
            raise ValueError(f"At least 7 valid images required for class {label}")
        random.Random(seed).shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        if not n_val or n - n_train - n_val < 1:
            raise ValueError(f"Not enough examples for three splits: {label}")
        for idx, row in enumerate(items):
            split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
            result.append({**row, "split": split})
    return sorted(result, key=lambda r: (r["split"], r["label"], r["path"]))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(root: Path, output: Path, seed: int = 42) -> dict:
    valid, rejected = inventory(root)
    manifest = stratified_split(valid, seed=seed)
    write_csv(output / "splits.csv", manifest, ["path", "label", "sha256", "split"])
    write_csv(output / "rejected.csv", rejected, ["path", "reason"])
    return {"accepted": len(manifest), "rejected": len(rejected),
            "by_split": dict(Counter(r["split"] for r in manifest)),
            "by_class": dict(Counter(r["label"] for r in manifest))}
