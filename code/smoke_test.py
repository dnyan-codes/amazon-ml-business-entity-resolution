from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from features import build_pair_labels, generate_pair_features, load_ground_truth_map
from model import EntityMatcherModel, build_ground_truth_map, build_prediction_map, evaluate_entity_level, threshold_search


RANDOM_STATE = 42


def build_smoke_fixture() -> Tuple[pd.DataFrame, List[Tuple[str, str, int]]]:
    """Build a tiny real-data smoke-test fixture with guaranteed positives.

    The fixture uses real records from the repo's training data and guarantees at
    least one known positive pair from train_ground_truth.tsv while keeping a few
    negative candidates. This is only a pipeline smoke test and is not used for
    the actual competition training or evaluation.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    s1_path = os.path.join(root, "data", "train_source1.tsv")
    gt_path = os.path.join(root, "data", "train_ground_truth.tsv")
    s2_path = os.path.join(root, "dataset", "train", "train_source2.tsv")
    s3_path = os.path.join(root, "dataset", "train", "train_source3.tsv")

    if not os.path.exists(s1_path) or not os.path.exists(gt_path):
        raise FileNotFoundError("Smoke test requires the Member 3 training split files in data/.")

    s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)

    if s1.empty or gt.empty:
        raise ValueError("Smoke fixture could not load the real Source 1 or ground-truth rows from the repo.")

    source1_map = {str(row["entity_id"]).strip(): row for row in s1.to_dict("records")}
    gt_map = load_ground_truth_map(gt_path)

    candidate_lookup: Dict[str, dict] = {}
    for source_path in (s2_path, s3_path):
        with open(source_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                candidate_id = str(row.get("entity_id", "")).strip()
                if candidate_id:
                    candidate_lookup[candidate_id] = row

    if not candidate_lookup:
        raise ValueError("No real candidate rows were found in the Source 2/3 training data.")

    selected_candidate_rows: List[Tuple[str, str]] = []
    for _, gt_row in gt.iterrows():
        source1_id = str(gt_row["source1_entity_id"]).strip()
        if source1_id not in source1_map:
            continue

        true_ids = [item.strip() for item in str(gt_row["matched_entity_ids"]).split(",") if item.strip()]
        valid_positive_ids = [candidate_id for candidate_id in true_ids if candidate_id in candidate_lookup]
        if not valid_positive_ids:
            continue

        positive_id = valid_positive_ids[0]
        negative_ids = [candidate_id for candidate_id in candidate_lookup if candidate_id not in set(true_ids)][:2]
        if len(negative_ids) < 2:
            continue

        selected_candidate_rows = [(source1_id, positive_id), (source1_id, negative_ids[0]), (source1_id, negative_ids[1])]
        break

    if not selected_candidate_rows:
        raise ValueError("No valid Source 1 entity with a real positive candidate match could be selected from the training ground truth.")

    labels = build_pair_labels(selected_candidate_rows, gt_map)
    positives = sum(1 for _, _, label in labels if label == 1)
    negatives = sum(1 for _, _, label in labels if label == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("Smoke fixture must include both positive and negative candidate pairs from real records.")

    row_records = []
    for source1_id, candidate_id in selected_candidate_rows:
        source1_row = source1_map[source1_id]
        candidate_row = candidate_lookup.get(candidate_id)
        if candidate_row is None:
            continue
        feature_dict = generate_pair_features(source1_row, candidate_row)
        feature_dict["source1_entity_id"] = source1_id
        feature_dict["candidate_entity_id"] = candidate_id
        row_records.append(feature_dict)

    if not row_records:
        raise ValueError("No feature rows were generated from the selected real candidate pairs.")

    feature_frame = pd.DataFrame(row_records)
    labels_df = pd.DataFrame(labels, columns=["source1_entity_id", "candidate_entity_id", "label"])
    feature_frame = feature_frame.merge(labels_df, on=["source1_entity_id", "candidate_entity_id"], how="inner")

    if feature_frame.empty:
        raise ValueError("Smoke test did not produce any valid feature rows after joining labels.")

    return feature_frame, labels


def run_smoke_test() -> None:
    print("========================================")
    print("MEMBER 2 SMOKE TEST")
    print("========================================")
    feature_frame, labels = build_smoke_fixture()

    source1_entities = feature_frame["source1_entity_id"].nunique()
    candidate_pairs = len(feature_frame)
    positive_pairs = int((feature_frame["label"] == 1).sum())
    negative_pairs = int((feature_frame["label"] == 0).sum())

    print(f"Source 1 entities: {source1_entities}")
    print(f"Candidate pairs: {candidate_pairs}")
    print(f"Positive pairs: {positive_pairs}")
    print(f"Negative pairs: {negative_pairs}")

    feature_columns = [col for col in feature_frame.columns if col not in {"source1_entity_id", "candidate_entity_id", "label"}]
    print(f"Feature count: {len(feature_columns)}")

    feature_frame = feature_frame[feature_columns + ["source1_entity_id", "candidate_entity_id", "label"]]
    if feature_frame.isna().sum().sum() > 0:
        print(f"NaN values: {feature_frame.isna().sum().sum()}")
    else:
        print("NaN values: 0")

    if not np.isfinite(feature_frame[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains non-finite values.")

    model = EntityMatcherModel(random_state=RANDOM_STATE)
    model.fit(feature_frame, feature_frame["label"].tolist())
    probs = model.predict_proba(feature_frame)
    if probs.shape[0] != len(feature_frame):
        raise ValueError("Probability prediction output is not aligned with the feature frame.")

    model.threshold = 0.5
    pred_map = build_prediction_map(feature_frame.assign(probability=probs), probability_col="probability", threshold=model.threshold)
    gt_map = build_ground_truth_map(feature_frame)
    evaluation = evaluate_entity_level(pred_map, gt_map)
    print("Model fit: PASS")
    print("Prediction: PASS")
    print("Thresholding: PASS")
    print("F0.5 evaluation: PASS")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F0.5: {evaluation['f05']:.4f}")
    print("========================================")
    print("SMOKE TEST COMPLETE")
    print("========================================")
    print("These metrics are for pipeline verification only.")
    print("They are NOT competition results.")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:  # pragma: no cover - smoke test should be explicit
        print("SMOKE TEST FAILED")
        print(str(exc))
        raise
