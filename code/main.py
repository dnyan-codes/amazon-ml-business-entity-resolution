from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import tempfile
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from features import (
    FEATURE_COLUMNS,
    PreparedRecord,
    generate_pair_feature_values,
    parse_matched_ids,
    prepare_normalized_record,
    prepare_record,
)
from model import (
    DEFAULT_THRESHOLD_GRID,
    EntityMatcherModel,
    compute_f05,
    compute_precision,
    compute_recall,
)


SOURCE_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
GROUND_TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}
CANDIDATE_COLUMNS = {"source1_entity_id", "candidate_entity_ids"}
BATCH_SIZE = 10_000
HIST_GRADIENT_BOOSTING_SAMPLE_SIZE = 500_000


def check_header(path: str, required: set[str]) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required input file not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"), [])
    missing = required - set(header)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def iter_candidate_rows(path: str) -> Iterator[Tuple[str, List[str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not CANDIDATE_COLUMNS.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Candidate file {path} must contain {sorted(CANDIDATE_COLUMNS)}")
        for row in reader:
            source1_id = str(row.get("source1_entity_id", "")).strip()
            if not source1_id:
                raise ValueError(f"Candidate file {path} contains a blank Source 1 ID.")
            raw_ids = str(row.get("candidate_entity_ids", "") or "")
            candidate_ids = list(dict.fromkeys(value.strip() for value in raw_ids.split(",") if value.strip()))
            yield source1_id, candidate_ids


def _insert_source(conn: sqlite3.Connection, path: str, split: str, source: str) -> int:
    check_header(path, SOURCE_COLUMNS)
    count = 0
    batch = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = str(row.get("entity_id", "")).strip()
            if not entity_id:
                raise ValueError(f"Blank entity_id found in {path}.")
            prepared = prepare_record(row)
            batch.append((entity_id, split, source,
                          prepared.name, prepared.address, prepared.country,
                          " ".join(prepared.name_tokens),
                          " ".join(prepared.address_tokens)))
            count += 1
            if len(batch) >= BATCH_SIZE:
                conn.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
    if batch:
        conn.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
    return count


def _insert_ground_truth(conn: sqlite3.Connection, path: str) -> int:
    check_header(path, GROUND_TRUTH_COLUMNS)
    count = 0
    batch = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source1_id = str(row.get("source1_entity_id", "")).strip()
            if not source1_id:
                continue
            for candidate_id in parse_matched_ids(row.get("matched_entity_ids")):
                batch.append((source1_id, candidate_id))
                count += 1
            if len(batch) >= BATCH_SIZE:
                conn.executemany("INSERT OR IGNORE INTO truth VALUES (?, ?)", batch)
                batch.clear()
    if batch:
        conn.executemany("INSERT OR IGNORE INTO truth VALUES (?, ?)", batch)
    return count


def _validate_candidate_file(conn: sqlite3.Connection, path: str, split: str) -> Tuple[int, int]:
    seen_table = "seen_" + split
    conn.execute(f"CREATE TABLE {seen_table} (entity_id TEXT PRIMARY KEY)")
    rows = pairs = 0
    for source1_id, candidate_ids in iter_candidate_rows(path):
        record = conn.execute(
            "SELECT split, source FROM records WHERE entity_id = ?", (source1_id,)
        ).fetchone()
        if record is None or record[0] != split or record[1] != "source1":
            raise ValueError(f"Candidate file {path} has an unknown or wrong-split Source 1 ID: {source1_id}")
        try:
            conn.execute(f"INSERT INTO {seen_table} VALUES (?)", (source1_id,))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Candidate file {path} repeats Source 1 row {source1_id}.") from exc
        for candidate_id in candidate_ids:
            if candidate_id.startswith("S1-") or not candidate_id.startswith(("S2-", "S3-")):
                raise ValueError(f"Invalid candidate ID prefix in {path}: {candidate_id}")
            candidate = conn.execute(
                "SELECT source FROM records WHERE entity_id = ?", (candidate_id,)
            ).fetchone()
            expected_source = "source2" if candidate_id.startswith("S2-") else "source3"
            if candidate is None or candidate[0] != expected_source:
                raise ValueError(f"Candidate ID {candidate_id} in {path} is absent from its source pool.")
        rows += 1
        pairs += len(candidate_ids)
    expected = conn.execute(
        "SELECT COUNT(*) FROM records WHERE split = ? AND source = 'source1'", (split,)
    ).fetchone()[0]
    missing = conn.execute(
        f"SELECT entity_id FROM records WHERE split = ? AND source = 'source1' "
        f"AND entity_id NOT IN (SELECT entity_id FROM {seen_table}) LIMIT 1", (split,)
    ).fetchone()
    if rows != expected or missing:
        raise ValueError(f"{path} has {rows:,} Source 1 rows; expected exactly {expected:,} for split {split}.")
    return rows, pairs


def _get_prepared_records(conn: sqlite3.Connection, entity_ids: Sequence[str]) -> Dict[str, PreparedRecord]:
    records: Dict[str, PreparedRecord] = {}
    unique_ids = list(dict.fromkeys(entity_ids))
    for offset in range(0, len(unique_ids), 900):
        batch_ids = unique_ids[offset:offset + 900]
        placeholders = ",".join("?" for _ in batch_ids)
        query = (
            "SELECT entity_id, name_norm, address_norm, country_norm, name_tokens, address_tokens "
            f"FROM records WHERE entity_id IN ({placeholders})"
        )
        for row in conn.execute(query, batch_ids):
            records[row[0]] = prepare_normalized_record(*row[1:])
    if len(records) != len(unique_ids):
        missing = set(unique_ids) - records.keys()
        raise ValueError(f"Entity record not found for candidate ID {next(iter(missing))}.")
    return records


def _iter_candidate_pair_chunks(
    candidate_path: str, chunk_size: int = BATCH_SIZE
) -> Iterator[List[Tuple[str, str]]]:
    batch: List[Tuple[str, str]] = []
    for source1_id, candidate_ids in iter_candidate_rows(candidate_path):
        for candidate_id in candidate_ids:
            batch.append((source1_id, candidate_id))
            if len(batch) >= chunk_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _positive_pairs_for_batch(
    conn: sqlite3.Connection, pair_batch: Sequence[Tuple[str, str]]
) -> set[Tuple[str, str]]:
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS pair_batch "
        "(source1_id TEXT, candidate_id TEXT, PRIMARY KEY(source1_id, candidate_id)) WITHOUT ROWID"
    )
    conn.execute("DELETE FROM pair_batch")
    conn.executemany("INSERT INTO pair_batch VALUES (?, ?)", pair_batch)
    return set(conn.execute(
        "SELECT pair_batch.source1_id, pair_batch.candidate_id "
        "FROM pair_batch JOIN truth USING (source1_id, candidate_id)"
    ))


def _count_candidate_labels(conn: sqlite3.Connection, candidate_path: str) -> Tuple[int, int]:
    pair_count = positive_count = 0
    for pair_batch in _iter_candidate_pair_chunks(candidate_path):
        positive_count += len(_positive_pairs_for_batch(conn, pair_batch))
        pair_count += len(pair_batch)
    return pair_count, positive_count


def _iter_feature_batches(
    conn: sqlite3.Connection, candidate_path: str
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for pair_batch in _iter_candidate_pair_chunks(candidate_path):
        positive_pairs = _positive_pairs_for_batch(conn, pair_batch)
        records = _get_prepared_records(
            conn, [entity_id for pair in pair_batch for entity_id in pair]
        )
        features = np.empty((len(pair_batch), len(FEATURE_COLUMNS)), dtype=np.float32)
        labels = np.zeros(len(pair_batch), dtype=np.uint8)
        for row_index, (source1_id, candidate_id) in enumerate(pair_batch):
            features[row_index] = generate_pair_feature_values(
                records[source1_id], records[candidate_id]
            )
            labels[row_index] = (source1_id, candidate_id) in positive_pairs
        yield features, labels


def _update_reservoir(
    sample_features: np.ndarray,
    sample_labels: np.ndarray,
    sample_count: int,
    rows_seen: int,
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    row_count = len(features)
    initial_count = min(row_count, len(sample_features) - sample_count)
    if initial_count:
        sample_features[sample_count:sample_count + initial_count] = features[:initial_count]
        sample_labels[sample_count:sample_count + initial_count] = labels[:initial_count]
        sample_count += initial_count

    replacement_count = row_count - initial_count
    if replacement_count:
        global_rows = rows_seen + initial_count + np.arange(replacement_count)
        slots = rng.integers(0, global_rows + 1, dtype=np.int64)
        selected = np.flatnonzero(slots < len(sample_features))
        for selected_index in selected:
            slot = slots[selected_index]
            source_index = initial_count + selected_index
            sample_features[slot] = features[source_index]
            sample_labels[slot] = labels[source_index]
    return sample_count, rows_seen + row_count


def _write_feature_matrix(
    conn: sqlite3.Connection,
    candidate_path: str,
    pair_count: int,
    work_dir: str,
    matrix_name: str,
) -> Tuple[np.memmap, np.memmap, List[str], int]:
    feature_columns = list(FEATURE_COLUMNS)
    X = np.memmap(os.path.join(work_dir, f"{matrix_name}_features.dat"), mode="w+", dtype=np.float32,
                  shape=(pair_count, len(feature_columns)))
    y = np.memmap(os.path.join(work_dir, f"{matrix_name}_labels.dat"), mode="w+", dtype=np.uint8,
                  shape=(pair_count,))
    row_index = positives = 0
    for feature_batch, label_batch in _iter_feature_batches(conn, candidate_path):
        end_index = row_index + len(feature_batch)
        X[row_index:end_index] = feature_batch
        y[row_index:end_index] = label_batch
        positives += int(np.sum(label_batch, dtype=np.uint64))
        row_index = end_index
    if row_index != pair_count:
        raise RuntimeError(f"Candidate pair count changed while reading {candidate_path}.")
    X.flush()
    y.flush()
    return X, y, feature_columns, positives


def _validation_thresholds(
    conn: sqlite3.Connection,
    candidate_path: str,
    model: EntityMatcherModel,
    validation_features: np.memmap,
    validation_labels: np.memmap,
) -> Dict[float, Dict[str, float | int]]:
    accum = {threshold: {"f05": 0.0, "precision": 0.0, "recall": 0.0,
                         "predicted_matches": 0, "tp": 0, "fp": 0, "fn": 0}
             for threshold in DEFAULT_THRESHOLD_GRID}
    thresholds = np.asarray(DEFAULT_THRESHOLD_GRID, dtype=np.float32)
    entity_count = 0
    candidate_pair_count = 0
    positive_candidate_count = 0
    ground_truth_match_count = 0
    pair_offset = 0
    for source1_id, candidate_ids in iter_candidate_rows(candidate_path):
        candidate_pair_count += len(candidate_ids)
        true_ids = {row[0] for row in conn.execute(
            "SELECT candidate_id FROM truth WHERE source1_id = ?", (source1_id,)
        )}
        ground_truth_match_count += len(true_ids)
        entity_count += 1
        predicted_counts = np.zeros(len(thresholds), dtype=np.int64)
        true_positive_counts = np.zeros(len(thresholds), dtype=np.int64)
        for start in range(0, len(candidate_ids), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(candidate_ids))
            matrix_start = pair_offset + start
            matrix_end = pair_offset + end
            probabilities = model.predict_proba_array(
                validation_features[matrix_start:matrix_end]
            )
            positive_candidate_count += int(
                np.sum(validation_labels[matrix_start:matrix_end], dtype=np.uint64)
            )
            truth_mask = np.fromiter(
                (candidate_id in true_ids for candidate_id in candidate_ids[start:end]),
                dtype=np.bool_,
                count=end - start,
            )
            predicted_mask = probabilities[:, None] >= thresholds[None, :]
            predicted_counts += np.sum(predicted_mask, axis=0, dtype=np.int64)
            true_positive_counts += np.sum(
                predicted_mask & truth_mask[:, None], axis=0, dtype=np.int64
            )

        pair_offset += len(candidate_ids)
        for index, threshold in enumerate(DEFAULT_THRESHOLD_GRID):
            totals = accum[threshold]
            tp = int(true_positive_counts[index])
            predicted_count = int(predicted_counts[index])
            fp = predicted_count - tp
            fn = len(true_ids) - tp
            if not true_ids and not predicted_count:
                f05 = precision = recall = 1.0
            else:
                precision = compute_precision(tp, fp)
                recall = compute_recall(tp, fn)
                f05 = compute_f05(precision, recall)
            totals["f05"] += f05
            totals["precision"] += precision
            totals["recall"] += recall
            totals["predicted_matches"] += predicted_count
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
    if not entity_count:
        raise ValueError("Validation candidate file contains no entities.")
    if pair_offset != len(validation_labels):
        raise RuntimeError("Validation candidate count changed while scoring the feature matrix.")
    results = {}
    for threshold, totals in accum.items():
        results[threshold] = {
            "entities_scored": entity_count,
            "candidate_pairs": candidate_pair_count,
            "positive_candidate_pairs": positive_candidate_count,
            "candidate_recall": (positive_candidate_count / ground_truth_match_count
                                 if ground_truth_match_count else 0.0),
            "predicted_matches": int(totals["predicted_matches"]),
            "precision": float(totals["precision"] / entity_count),
            "recall": float(totals["recall"] / entity_count),
            "macro_f05": float(totals["f05"] / entity_count),
            "tp": int(totals["tp"]), "fp": int(totals["fp"]), "fn": int(totals["fn"]),
        }
    return results


def _predict_test(
    conn: sqlite3.Connection,
    candidate_path: str,
    output_path: str,
    model: EntityMatcherModel,
    feature_columns: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for source1_id, candidate_ids in iter_candidate_rows(candidate_path):
            if candidate_ids:
                records = _get_prepared_records(conn, [source1_id, *candidate_ids])
                features = np.asarray(
                    [
                        generate_pair_feature_values(records[source1_id], records[candidate_id])
                        for candidate_id in candidate_ids
                    ],
                    dtype=np.float32,
                )
                probabilities = model.predict_proba_array(features)
                matched = [candidate_ids[i] for i, value in enumerate(probabilities) if value >= model.threshold]
            else:
                matched = []
            writer.writerow([source1_id, ",".join(matched)])
def _create_store(args: argparse.Namespace, database_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("CREATE TABLE records (entity_id TEXT PRIMARY KEY, split TEXT, source TEXT, name_norm TEXT, address_norm TEXT, country_norm TEXT, name_tokens TEXT, address_tokens TEXT)")
    conn.execute("CREATE TABLE truth (source1_id TEXT, candidate_id TEXT, PRIMARY KEY(source1_id, candidate_id))")
    for path, split in ((args.train_s1, "train"), (args.val_s1, "val")):

        print(f"Indexing {path}...")
        _insert_source(conn, path, split, "source1")
    if args.infer_test:
        print(f"Indexing {args.test_s1}...")
        _insert_source(conn, args.test_s1, "test", "source1")
    for path, source in ((args.train_s2, "source2"), (args.train_s3, "source3")):
        print(f"Indexing {path}...")
        _insert_source(conn, path, "pool", source)
    if args.infer_test:
        for path, source in ((args.test_s2, "source2"), (args.test_s3, "source3")):
            print(f"Indexing {path}...")
            _insert_source(conn, path, "test_pool", source)
    _insert_ground_truth(conn, args.train_gt)
    _insert_ground_truth(conn, args.val_gt)
    conn.commit()
    return conn


def _dry_run(args: argparse.Namespace) -> None:
    source_files = [args.train_s1, args.val_s1, args.train_s2, args.train_s3, args.train_gt, args.val_gt]
    if args.infer_test:
        source_files += [args.test_s1, args.test_s2, args.test_s3]
    for path in source_files:
        required = GROUND_TRUTH_COLUMNS if path in {args.train_gt, args.val_gt} else SOURCE_COLUMNS
        check_header(path, required)
    candidate_files = [(args.train_candidates, "train"), (args.val_candidates, "val")]
    if args.infer_test:
        candidate_files.append((args.test_candidates, "test"))
    for path, _ in candidate_files:
        check_header(path, CANDIDATE_COLUMNS)
        rows = pairs = 0
        for _, candidates in iter_candidate_rows(path):
            rows += 1
            pairs += len(candidates)
        print(f"{path}: {rows:,} Source 1 rows; {pairs:,} candidate pairs")
    print("Dry run passed file/schema checks; no dataset rows were loaded into memory and no model was trained.")


def run(args: argparse.Namespace) -> None:
    if args.dry_run:
        _dry_run(args)
        return
    for path in (args.train_candidates, args.val_candidates):
        check_header(path, CANDIDATE_COLUMNS)
    with tempfile.TemporaryDirectory(prefix="entity_matcher_") as work_dir:
        conn = _create_store(args, os.path.join(work_dir, "records.sqlite"))
        validation_X = validation_y = None
        try:
            train_rows, train_pairs = _validate_candidate_file(conn, args.train_candidates, "train")
            val_rows, val_pairs = _validate_candidate_file(conn, args.val_candidates, "val")
            print(f"Training: {train_rows:,} entities / {train_pairs:,} pairs; validation: {val_rows:,} entities / {val_pairs:,} pairs")
            if train_pairs == 0 or val_pairs == 0:
                raise ValueError("Training and validation candidate files must contain at least one pair.")
            label_pair_count, positive_pairs = _count_candidate_labels(
                conn, args.train_candidates
            )
            if label_pair_count != train_pairs:
                raise RuntimeError("Training candidate count changed while counting labels.")
            if positive_pairs == 0 or positive_pairs == train_pairs:
                raise ValueError("Training candidates must contain both positive and negative labels.")
            feature_columns = list(FEATURE_COLUMNS)
            validation_X, validation_y, validation_columns, _ = _write_feature_matrix(
                conn, args.val_candidates, val_pairs, work_dir, "validation"
            )
            if validation_columns != feature_columns:
                raise RuntimeError("Training and validation feature columns differ.")
            print(f"Training labels: {positive_pairs:,} positive / {train_pairs - positive_pairs:,} negative")

            class_weights = {
                0: train_pairs / (2 * (train_pairs - positive_pairs)),
                1: train_pairs / (2 * positive_pairs),
            }
            logistic_model = EntityMatcherModel(
                random_state=args.seed, model_type="logistic"
            )
            logistic_model.feature_columns = feature_columns
            logistic_model.begin_incremental_fit(class_weights)

            sample_capacity = min(HIST_GRADIENT_BOOSTING_SAMPLE_SIZE, train_pairs)
            sample_features = np.empty(
                (sample_capacity, len(feature_columns)), dtype=np.float32
            )
            sample_labels = np.empty(sample_capacity, dtype=np.uint8)
            sample_count = rows_seen = streamed_positive_pairs = 0
            rng = np.random.default_rng(args.seed)
            for feature_batch, label_batch in _iter_feature_batches(
                conn, args.train_candidates
            ):
                logistic_model.partial_fit_array(feature_batch, label_batch)
                sample_count, rows_seen = _update_reservoir(
                    sample_features,
                    sample_labels,
                    sample_count,
                    rows_seen,
                    feature_batch,
                    label_batch,
                    rng,
                )
                streamed_positive_pairs += int(
                    np.sum(label_batch, dtype=np.uint64)
                )
            if rows_seen != train_pairs or streamed_positive_pairs != positive_pairs:
                raise RuntimeError("Training candidate labels changed while streaming features.")

            hist_model = EntityMatcherModel(
                random_state=args.seed, model_type="hist_gradient_boosting"
            ).fit_array(
                sample_features[:sample_count],
                sample_labels[:sample_count],
                feature_columns,
            )
            print(
                f"HistGradientBoosting comparison trained on a uniform "
                f"reservoir sample of {sample_count:,} / {train_pairs:,} pairs."
            )

            results = []
            threshold_sweeps = {}
            for model in (logistic_model, hist_model):
                model_type = model.model_type
                print(f"Scoring {model_type} on the complete validation candidate set...")
                threshold_results = _validation_thresholds(
                    conn, args.val_candidates, model, validation_X, validation_y
                )
                threshold_sweeps[model_type] = {
                    f"{threshold:.2f}": metrics
                    for threshold, metrics in threshold_results.items()
                }
                best_threshold = max(threshold_results, key=lambda threshold: (threshold_results[threshold]["macro_f05"], threshold))
                metrics = threshold_results[best_threshold]
                metrics["threshold"] = best_threshold
                metrics["training_positive_pairs"] = positive_pairs
                metrics["training_candidate_pairs"] = (
                    train_pairs if model_type == "logistic" else sample_count
                )
                model.threshold = best_threshold
                model.validation_metrics = metrics
                print(
                    f"{model_type}: threshold={best_threshold:.2f}, "
                    f"TP={metrics['tp']:,}, FP={metrics['fp']:,}, FN={metrics['fn']:,}, "
                    f"precision={metrics['precision']:.5f}, recall={metrics['recall']:.5f}, "
                    f"macro F0.5={metrics['macro_f05']:.5f}, "
                    f"candidate recall={metrics['candidate_recall']:.5f}"
                )
                model.save(os.path.join(args.model_dir, model_type))
                results.append((float(metrics["macro_f05"]), model_type == "logistic", model, metrics))
            _, _, selected_model, selected_metrics = max(results, key=lambda result: (result[0], result[1]))
            selected_model.save(args.model_dir)
            comparison_path = os.path.join(args.model_dir, "model_comparison.json")
            with open(comparison_path, "w", encoding="utf-8") as comparison_file:
                json.dump(
                    {
                        model.model_type: {
                            "threshold": model.threshold,
                            "best_validation": model.validation_metrics,
                            "threshold_sweep": threshold_sweeps[model.model_type],
                        }
                        for _, _, model, _ in results
                    },
                    comparison_file,
                    indent=2,
                )
            print(f"Selected {selected_model.model_type}; saved artifacts under {args.model_dir}")
            print(f"Saved both validation results to {comparison_path}")
            print(
                f"Selected-model validation: entities={selected_metrics['entities_scored']:,}, "
                f"pairs={selected_metrics['candidate_pairs']:,}, "
                f"positive candidates={selected_metrics['positive_candidate_pairs']:,}, "
                f"candidate recall={selected_metrics['candidate_recall']:.5f}, "
                f"predicted={selected_metrics['predicted_matches']:,}, "
                f"threshold={selected_model.threshold:.2f}"
            )
            if args.infer_test:
                _validate_candidate_file(conn, args.test_candidates, "test")
                _predict_test(conn, args.test_candidates, args.matching_output, selected_model, feature_columns)
                print(f"Wrote matching results to {args.matching_output}")
        finally:
            conn.close()
            for matrix in (validation_X, validation_y):
                if isinstance(matrix, np.memmap):
                    matrix.flush()
                    matrix._mmap.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Member 2 entity matcher.")
    parser.add_argument("--train-s1", default="data/train_source1.tsv")
    parser.add_argument("--train-gt", default="data/train_ground_truth.tsv")
    parser.add_argument("--val-s1", default="data/val_source1.tsv")
    parser.add_argument("--val-gt", default="data/val_ground_truth.tsv")
    parser.add_argument("--train-s2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--train-s3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--test-s1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--test-s2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--test-s3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--train-candidates", default="output/train_candidate_pairs.tsv")
    parser.add_argument("--val-candidates", default="output/val_candidate_pairs.tsv")
    parser.add_argument("--test-candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--matching-output", default="output/matching_results.tsv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Check files, schemas, and candidate counts only.")
    parser.add_argument("--infer-test", action="store_true", help="After validation selection, run test inference.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())