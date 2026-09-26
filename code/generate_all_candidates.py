from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
MEMBER1_SRC_DIR = CODE_DIR / "business_entity_resolution" / "src"

if str(MEMBER1_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER1_SRC_DIR))

from blocking import generate_candidate_pairs
from preprocessing import load_source


SPLIT_FILES: Dict[str, Tuple[Path, Path, Path, Path]] = {
    "train": (
        PROJECT_ROOT / "data" / "train_source1.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
        PROJECT_ROOT / "output" / "train_candidate_pairs.tsv",
    ),
    "val": (
        PROJECT_ROOT / "data" / "val_source1.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
        PROJECT_ROOT / "output" / "val_candidate_pairs.tsv",
    ),
    "test": (
        PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv",
        PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv",
        PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv",
        PROJECT_ROOT / "output" / "candidate_pairs.tsv",
    ),
}

MAX_CANDIDATES = 100


def generate_for_split(split: str) -> Path:
    source1_path, source2_path, source3_path, output_path = SPLIT_FILES[split]
    input_paths = (source1_path, source2_path, source3_path)
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Cannot generate {split} candidates; input file(s) not found:\n{missing}")

    print(f"Generating candidate pairs for split: {split}")
    print(f"Source 1: {source1_path}")
    print(f"Source 2 pool: {source2_path}")
    print(f"Source 3 pool: {source3_path}")

    print("Loading and normalizing Source 1...")
    source1 = load_source(str(source1_path))
    print(f"Loaded {len(source1):,} Source 1 entities.")

    print("Loading and normalizing Source 2...")
    source2 = load_source(str(source2_path))
    print(f"Loaded {len(source2):,} Source 2 entities.")

    print("Loading and normalizing Source 3...")
    source3 = load_source(str(source3_path))
    print(f"Loaded {len(source3):,} Source 3 entities.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing candidates with max_candidates={MAX_CANDIDATES}...")
    generate_candidate_pairs(
        source1,
        source2,
        source3,
        str(output_path),
        max_candidates=MAX_CANDIDATES,
    )
    print(f"Finished {split} candidate generation: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Member 1 candidate pairs for exactly one data split."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=tuple(SPLIT_FILES),
        help="Select one split to generate: train, val, or test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_for_split(args.split)


if __name__ == "__main__":
    main()
