from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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
    if not true_ids and not pred_ids:
        return 1.0, 1.0, 1.0, 0, 0, 0
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
    entity_precisions: List[float] = []
    entity_recalls: List[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for source1_id in all_ids:
        true_ids = set(ground_truth_map.get(source1_id, set()))
        pred_ids = set(prediction_map.get(source1_id, set()))
        f05, precision, recall, tp, fp, fn = entity_level_f05(true_ids, pred_ids)
        entity_scores.append(f05)
        entity_precisions.append(precision)
        entity_recalls.append(recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f05 = float(sum(entity_scores) / len(entity_scores)) if entity_scores else 0.0
    macro_precision = float(sum(entity_precisions) / len(entity_precisions)) if entity_precisions else 0.0
    macro_recall = float(sum(entity_recalls) / len(entity_recalls)) if entity_recalls else 0.0
    total_precision = compute_precision(total_tp, total_fp)
    total_recall = compute_recall(total_tp, total_fn)

    return {
        "entities_scored": len(all_ids),
        "macro_f05": macro_f05,
        "precision": macro_precision,
        "recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_precision": total_precision,
        "micro_recall": total_recall,
        "f05": macro_f05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


class EntityMatcherModel:
    def __init__(self, random_state: int = 42, model_type: str = "logistic"):
        self.random_state = random_state
        if model_type not in {"logistic", "hist_gradient_boosting"}:
            raise ValueError("model_type must be 'logistic' or 'hist_gradient_boosting'.")
        self.model_type = model_type
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.model = self._new_model()
        self.threshold = 0.5
        self.validation_metrics: Dict[str, float | int] = {}
        self._partial_fit_initialized = False

    def _new_model(self):
        if self.model_type == "logistic":
            return SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=self.random_state,
                penalty="l2",
                alpha=0.0001,
                learning_rate="optimal",
                average=True,
                max_iter=1000,
                tol=1e-3,
            )
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=self.random_state,
            early_stopping=False,
        )

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "EntityMatcherModel":
        if X.empty:
            raise ValueError("Cannot fit on an empty feature matrix.")
        feature_columns = [col for col in X.columns if col not in {"source1_entity_id", "candidate_entity_id"}]
        if not feature_columns:
            raise ValueError("No feature columns found for model fitting.")
        X_selected = X[feature_columns]
        X_numeric = X_selected.to_numpy(dtype=np.float32, copy=False)
        return self.fit_array(X_numeric, y, feature_columns)

    def fit_array(
        self,
        X: np.ndarray,
        y: Sequence[int],
        feature_columns: Sequence[str],
    ) -> "EntityMatcherModel":
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty or non-2D feature matrix.")
        if X.shape[1] != len(feature_columns) or not feature_columns:
            raise ValueError("Feature matrix columns do not match feature_columns.")

        X_numeric = np.asarray(X, dtype=np.float32)
        np.nan_to_num(X_numeric, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_columns = list(feature_columns)

        y_array = np.asarray(y)
        if np.unique(y_array).size < 2:
            raise ValueError("Training data must contain both positive and negative labels for logistic regression.")

        if self.model_type == "logistic":
            self.scaler = None
            self.model.fit(
                self._scale_logistic_features(X_numeric, self.feature_columns), y_array
            )
        else:
            counts = np.bincount(y_array, minlength=2)
            sample_weights = np.where(y_array == 1, len(y_array) / (2 * max(counts[1], 1)), len(y_array) / (2 * max(counts[0], 1)))
            self.model.fit(X_numeric, y_array, sample_weight=sample_weights)
        return self

    @staticmethod
    def _scale_logistic_features(
        X: np.ndarray, feature_columns: Sequence[str] | None = None
    ) -> np.ndarray:
        scaled_columns = feature_columns or ()
        scales = np.ones(len(scaled_columns), dtype=np.float32)
        for index, column in enumerate(scaled_columns):
            if column.endswith(("_fuzz_ratio", "_partial_ratio", "_token_sort_ratio", "_token_set_ratio")):
                scales[index] = 0.01
            elif column in {"name_length_difference", "address_length_difference"}:
                scales[index] = 0.01
            elif column in {"name_token_count_difference", "address_token_count_difference"}:
                scales[index] = 0.1
        return np.multiply(X, scales, dtype=np.float32)

    def begin_incremental_fit(self, class_weights: Mapping[int, float]) -> None:
        if self.model_type != "logistic":
            raise ValueError("Incremental fitting is only available for the logistic model.")
        self.model = SGDClassifier(
            loss="log_loss",
            class_weight=dict(class_weights),
            random_state=self.random_state,
            penalty="l2",
            alpha=0.0001,
            learning_rate="optimal",
            average=True,
        )
        self.scaler = None
        self._partial_fit_initialized = False

    def partial_fit_array(self, X: np.ndarray, y: Sequence[int]) -> "EntityMatcherModel":
        if self.model_type != "logistic":
            raise ValueError("Incremental fitting is only available for the logistic model.")
        if not self.feature_columns:
            raise ValueError("Set feature_columns before incremental fitting.")
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ValueError("Feature matrix columns do not match the fitted model.")
        X_scaled = self._scale_logistic_features(X, self.feature_columns)
        if self._partial_fit_initialized:
            self.model.partial_fit(X_scaled, y)
        else:
            self.model.partial_fit(X_scaled, y, classes=np.asarray([0, 1], dtype=np.uint8))
            self._partial_fit_initialized = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns:
            raise ValueError("Model has not been fit yet.")
        X_selected = X[self.feature_columns]
        X_numeric = X_selected.to_numpy(dtype=np.float32, copy=False)
        return self.predict_proba_array(X_numeric)

    def predict_proba_array(self, X: np.ndarray) -> np.ndarray:
        if not self.feature_columns:
            raise ValueError("Model has not been fit yet.")
        if X.ndim != 2 or X.shape[1] != len(self.feature_columns):
            raise ValueError("Feature matrix columns do not match the fitted model.")
        X_numeric = np.asarray(X, dtype=np.float32)
        np.nan_to_num(X_numeric, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        values = X_numeric
        if self.scaler is not None:
            values = self.scaler.transform(values)
        elif self.model_type == "logistic":
            values = self._scale_logistic_features(values, self.feature_columns)
        return self.model.predict_proba(values)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.joblib"))
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
        elif os.path.exists(scaler_path):
            os.remove(scaler_path)
        with open(os.path.join(model_dir, "feature_columns.json"), "w", encoding="utf-8") as handle:
            json.dump(self.feature_columns, handle)
        with open(os.path.join(model_dir, "threshold.json"), "w", encoding="utf-8") as handle:
            json.dump({"threshold": float(self.threshold)}, handle)
        with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "model_type": self.model_type,
                "feature_names": self.feature_columns,
                "validation_threshold": float(self.threshold),
                **self.validation_metrics,
            }, handle, indent=2)

    def load(self, model_dir: str) -> "EntityMatcherModel":
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        with open(os.path.join(model_dir, "feature_columns.json"), "r", encoding="utf-8") as handle:
            self.feature_columns = json.load(handle)
        with open(os.path.join(model_dir, "threshold.json"), "r", encoding="utf-8") as handle:
            self.threshold = float(json.load(handle).get("threshold", 0.5))
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.model_type = metadata.get("model_type", self.model_type)
            self.validation_metrics = metadata
        return self


def threshold_search(
    model: EntityMatcherModel,
    X: pd.DataFrame,
    y: Sequence[int],
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    ground_truth_map: Optional[Mapping[str, set[str]]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Select the validation threshold maximizing macro Source-1 entity F0.5."""
    if "source1_entity_id" not in X.columns:
        raise ValueError("Threshold search requires a source1_entity_id column in the dataframe.")
    probs = model.predict_proba(X)
    X_eval = X.copy()
    X_eval["probability"] = probs
    X_eval["label"] = y

    best_threshold = float(threshold_grid[0])
    best_result: Dict[str, float | int] = {}
    for threshold in threshold_grid:
        pred_map: Dict[str, set[str]] = {}
        true_map = {key: set(value) for key, value in (ground_truth_map or {}).items()}
        grouped = X_eval.groupby("source1_entity_id")
        for source1_id, group in grouped:
            if ground_truth_map is None:
                true_map[source1_id] = set(group[group["label"] == 1]["candidate_entity_id"].tolist())
            pred_ids = set(group[group["probability"] >= threshold]["candidate_entity_id"].tolist())
            pred_map[source1_id] = pred_ids

        result = evaluate_entity_level(pred_map, true_map)
        if not best_result or result["macro_f05"] > best_result["macro_f05"]:
            best_threshold = float(threshold)
            best_result = result
    return best_threshold, {"best_threshold": best_threshold, **best_result}


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
