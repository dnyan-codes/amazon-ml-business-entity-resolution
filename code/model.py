from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler


DEFAULT_THRESHOLD_GRID = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
]


def compute_precision(tp: int, fp: int) -> float:
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def compute_recall(tp: int, fn: int) -> float:
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_f05(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_level_f05(true_ids: set[str], pred_ids: set[str]) -> Tuple[float, float, float, int, int, int]:
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)
    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    f05 = compute_f05(precision, recall)
    return f05, precision, recall, tp, fp, fn


def evaluate_entity_level(prediction_map: Mapping[str, set[str]], ground_truth_map: Mapping[str, set[str]]) -> Dict[str, float | int]:
    """Evaluate Source-1 entity-level predictions.

    Ground truth is treated as a mapping from source1_entity_id -> set(matched_ids).
    Predictions are treated similarly.
    """
    all_ids = sorted(set(ground_truth_map) | set(prediction_map))
    entity_scores: List[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for source1_id in all_ids:
        true_ids = set(ground_truth_map.get(source1_id, set()))
        pred_ids = set(prediction_map.get(source1_id, set()))
        f05, precision, recall, tp, fp, fn = entity_level_f05(true_ids, pred_ids)
        entity_scores.append(f05)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f05 = float(sum(entity_scores) / len(entity_scores)) if entity_scores else 0.0
    total_precision = compute_precision(total_tp, total_fp)
    total_recall = compute_recall(total_tp, total_fn)
    total_f05 = compute_f05(total_precision, total_recall)

    return {
        "entities_scored": len(all_ids),
        "macro_f05": macro_f05,
        "precision": total_precision,
        "recall": total_recall,
        "f05": total_f05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


class EntityMatcherModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_columns: List[str] = []
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=2000,
            solver="liblinear",
        )
        self.threshold = 0.5

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "EntityMatcherModel":
        if X.empty:
            raise ValueError("Cannot fit on an empty feature matrix.")
        feature_columns = [col for col in X.columns if col not in {"source1_entity_id", "candidate_entity_id"}]
        if not feature_columns:
            raise ValueError("No feature columns found for model fitting.")

        X_selected = X[feature_columns].copy()
        X_numeric = X_selected.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.feature_columns = feature_columns

        y_array = np.asarray(y, dtype=int)
        if np.unique(y_array).size < 2:
            raise ValueError("Training data must contain both positive and negative labels for logistic regression.")

        X_scaled = self.scaler.fit_transform(X_numeric.values)
        self.model.fit(X_scaled, y_array)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise ValueError("Model has not been fit yet.")
        X_selected = X[self.feature_columns].copy()
        X_numeric = X_selected.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_scaled = self.scaler.transform(X_numeric.values)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        with open(os.path.join(model_dir, "feature_columns.json"), "w", encoding="utf-8") as handle:
            json.dump(self.feature_columns, handle)
        with open(os.path.join(model_dir, "threshold.json"), "w", encoding="utf-8") as handle:
            json.dump({"threshold": float(self.threshold)}, handle)

    def load(self, model_dir: str) -> "EntityMatcherModel":
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        with open(os.path.join(model_dir, "feature_columns.json"), "r", encoding="utf-8") as handle:
            self.feature_columns = json.load(handle)
        with open(os.path.join(model_dir, "threshold.json"), "r", encoding="utf-8") as handle:
            self.threshold = float(json.load(handle).get("threshold", 0.5))
        return self


def threshold_search(
    model: EntityMatcherModel,
    X: pd.DataFrame,
    y: Sequence[int],
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> Tuple[float, Dict[str, float]]:
    """Select the validation threshold maximizing macro Source-1 entity F0.5."""
    if "source1_entity_id" not in X.columns:
        raise ValueError("Threshold search requires a source1_entity_id column in the dataframe.")
    scores: List[Tuple[float, float]] = []

    probs = model.predict_proba(X)
    X_eval = X.copy()
    X_eval["probability"] = probs
    X_eval["label"] = y

    for threshold in threshold_grid:
        pred_map: Dict[str, set[str]] = {}
        true_map: Dict[str, set[str]] = {}
        grouped = X_eval.groupby("source1_entity_id")
        for source1_id, group in grouped:
            true_ids = set(group[group["label"] == 1]["candidate_entity_id"].tolist())
            true_map[source1_id] = true_ids
            pred_ids = set(group[group["probability"] >= threshold]["candidate_entity_id"].tolist())
            pred_map[source1_id] = pred_ids

        result = evaluate_entity_level(pred_map, true_map)
        scores.append((threshold, float(result["macro_f05"])))

    best_threshold, best_score = max(scores, key=lambda item: item[1])
    return best_threshold, {"best_threshold": best_threshold, "best_macro_f05": best_score}


def build_prediction_map(frame: pd.DataFrame, probability_col: str = "probability", threshold: float = 0.5) -> Dict[str, set[str]]:
    prediction_map: Dict[str, set[str]] = {}
    for source1_id, group in frame.groupby("source1_entity_id"):
        pred_ids = set(group[group[probability_col] >= threshold]["candidate_entity_id"].tolist())
        prediction_map[source1_id] = pred_ids
    return prediction_map


def build_ground_truth_map(frame: pd.DataFrame) -> Dict[str, set[str]]:
    gt_map: Dict[str, set[str]] = {}
    for source1_id, group in frame.groupby("source1_entity_id"):
        gt_map[source1_id] = set(group[group["label"] == 1]["candidate_entity_id"].tolist())
    return gt_map
