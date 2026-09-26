from __future__ import annotations

import argparse
import os
from typing import Optional


def run_training(candidate_pairs_path: str, source1_path: str, ground_truth_path: str, output_dir: str) -> None:
    """Planned full Member 2 pipeline entry point.

    This function intentionally does not run automatically. Member 1 must provide
    a candidate-pair file first. Once it exists, the pipeline can be wired to:
      - read candidate pairs
      - generate pairwise features
      - build labels
      - train Logistic Regression
      - choose threshold on validation data
      - save model artifacts
      - predict on test candidates
    """
    if not os.path.exists(candidate_pairs_path):
        raise FileNotFoundError(
            "Member 1 candidate-pair file not found.\n"
            "Waiting for candidate generation from Member 1."
        )

    # Placeholder for the real full Member 2 pipeline.
    raise NotImplementedError("Full training pipeline is intentionally not executed in this inspection stage.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Member 2 pipeline entry point")
    parser.add_argument("--candidate-pairs", default="output/candidate_pairs.tsv", help="Path to the future Member 1 candidate-pair file.")
    parser.add_argument("--source1", default="data/train_source1.tsv", help="Source 1 training TSV")
    parser.add_argument("--ground-truth", default="data/train_ground_truth.tsv", help="Ground truth TSV for training")
    parser.add_argument("--output-dir", default="models", help="Directory for saved model artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.candidate_pairs, args.source1, args.ground_truth, args.output_dir)
