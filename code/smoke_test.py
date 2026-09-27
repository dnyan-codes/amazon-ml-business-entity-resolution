from __future__ import annotations

import numpy as np
import pandas as pd

from features import generate_pair_features, normalize_text
from model import EntityMatcherModel, entity_level_f05, threshold_search


def run_smoke_test() -> None:
    assert normalize_text("Ｃａｆé & 東京") == normalize_text("Café and 東京")
    assert normalize_text(float("nan")) == ""

    records = [
        ("s1-match", "S2-positive", "Café & Co", "12 Rue de Paris", "France", 1),
        ("s1-match", "S3-negative", "Cafe Holdings", "900 Market Street", "France", 0),
        ("s1-empty", "S2-negative", "Different Shop", "8 Main Road", "India", 0),
        ("s1-empty", "S3-negative", "Another Company", "77 Lake View", "US", 0),
    ]
    rows = []
    labels = []
    for source1_id, candidate_id, name, address, country, label in records:
        source1 = {
            "business_name": "Ｃａｆé and Co" if source1_id == "s1-match" else "No Listed Business",
            "business_address": "12 Rue de Paris" if source1_id == "s1-match" else "1 Empty Street",
            "country": country,
        }
        candidate = {"business_name": name, "business_address": address, "country": country}
        row = generate_pair_features(source1, candidate)
        row["source1_entity_id"] = source1_id
        row["candidate_entity_id"] = candidate_id
        rows.append(row)
        labels.append(label)

    frame = pd.DataFrame(rows)
    feature_columns = [column for column in frame if column not in {"source1_entity_id", "candidate_entity_id"}]
    assert frame.loc[0, "exact_normalized_name"] == 1
    assert frame.loc[0, "country_exact_match"] == 1
    assert np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all()
    assert sum(labels) == 1 and 0 in labels

    model = EntityMatcherModel(random_state=7).fit(frame, labels)
    probabilities = model.predict_proba(frame)
    assert len(probabilities) == len(frame)

    validation_truth = {"s1-match": {"S2-positive"}, "s1-empty": set()}
    threshold, metrics = threshold_search(
        model, frame, labels, ground_truth_map=validation_truth
    )
    assert 0.10 <= threshold <= 0.90
    assert metrics["entities_scored"] == 2
    assert np.isfinite(metrics["macro_f05"])
    assert entity_level_f05(set(), set())[0] == 1.0
    model.threshold = threshold
    predictions = (probabilities >= model.threshold).astype(int)
    assert predictions.shape == (len(frame),)
    print("SMOKE TEST PASS: Unicode features, labels, fit, prediction, and thresholding")


if __name__ == "__main__":
    run_smoke_test()