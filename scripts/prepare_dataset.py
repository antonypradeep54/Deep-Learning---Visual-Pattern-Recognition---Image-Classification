"""CLI for deterministic image inventory and split creation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare Cats vs Dogs dataset split manifests")
    parser.add_argument("--input", required=True, type=Path, help="Root directory of extracted images")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
