from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(ROOT_DIR, "data", "val_ground_truth.tsv")
CANDIDATE_FILES = (
    ("100-candidate validation", os.path.join(ROOT_DIR, "output", "val_candidate_pairs.tsv"), 100),
    ("250-candidate validation", os.path.join(ROOT_DIR, "output", "val_candidate_pairs_250.tsv"), 250),
    ("500-candidate validation", os.path.join(ROOT_DIR, "output", "val_candidate_pairs_500.tsv"), 500),
)


def parse_ids(raw_value: object) -> List[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def load_ground_truth(path: str) -> Dict[str, set[str]]:
    truth: Dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ground-truth file {path} is missing columns: {sorted(missing)}")
        for row in reader:
            source1_id = str(row.get("source1_entity_id", "") or "").strip()
            if not source1_id:
                continue
            matched_ids = set(parse_ids(row.get("matched_entity_ids", "")))
            truth[source1_id] = matched_ids
    return truth


def iter_candidate_rows(path: str) -> Iterable[Tuple[str, List[str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Candidate file {path} is missing columns: {sorted(missing)}")
        for row in reader:
            source1_id = str(row.get("source1_entity_id", "") or "").strip()
            if not source1_id:
                continue
            candidate_ids = list(dict.fromkeys(parse_ids(row.get("candidate_entity_ids", ""))))
            yield source1_id, candidate_ids


def compute_candidate_metrics(path: str, cap: int, ground_truth: Dict[str, set[str]]) -> Dict[str, float | int]:
    entity_count = 0
    total_candidate_pairs = 0
    captured_true_match_pairs = 0
    covered_entities = 0
    max_candidates_per_entity = 0
    hit_cap_count = 0
    seen_source1_ids = set()

    for source1_id, candidate_ids in iter_candidate_rows(path):
        if source1_id in seen_source1_ids:
            raise ValueError(f"Candidate file {path} contains a duplicate Source-1 entity ID: {source1_id}")
        seen_source1_ids.add(source1_id)

        entity_count += 1
        total_candidate_pairs += len(candidate_ids)
        max_candidates_per_entity = max(max_candidates_per_entity, len(candidate_ids))
        if len(candidate_ids) == cap:
            hit_cap_count += 1

        true_ids = ground_truth.get(source1_id, set())
        true_matches_captured = set(candidate_ids) & true_ids
        captured_true_match_pairs += len(true_matches_captured)
        if true_matches_captured:
            covered_entities += 1

    total_ground_truth_match_pairs = sum(len(ids) for ids in ground_truth.values())
    recall = (captured_true_match_pairs / total_ground_truth_match_pairs) if total_ground_truth_match_pairs else 0.0
    average_candidates = (total_candidate_pairs / entity_count) if entity_count else 0.0
    coverage_percent = (covered_entities / entity_count * 100.0) if entity_count else 0.0
    cap_hit_percent = (hit_cap_count / entity_count * 100.0) if entity_count else 0.0

    return {
        "validation_source1_entities": entity_count,
        "total_candidate_pairs": total_candidate_pairs,
        "total_ground_truth_match_pairs": total_ground_truth_match_pairs,
        "true_match_pairs_captured": captured_true_match_pairs,
        "candidate_pair_recall": recall,
        "source1_entities_with_at_least_one_true_match_captured": covered_entities,
        "source1_entity_coverage_percentage": coverage_percent,
        "average_candidates_per_source1_entity": average_candidates,
        "max_candidates_per_source1_entity": max_candidates_per_entity,
        "source1_entities_hitting_cap": hit_cap_count,
        "source1_entities_hitting_cap_percentage": cap_hit_percent,
    }


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def render_row(columns: Sequence[str]) -> str:
        return " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(columns))

    table_lines = [render_row(headers), "-+-".join("-" * width for width in widths)]
    for row in rows:
        table_lines.append(render_row([str(value) for value in row]))
    return "\n".join(table_lines)


def main() -> None:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    results: Dict[str, Dict[str, float | int]] = {}
    for label, path, cap in CANDIDATE_FILES:
        results[label] = compute_candidate_metrics(path, cap, ground_truth)

    recall_100 = float(results["100-candidate validation"]["candidate_pair_recall"])
    recall_250 = float(results["250-candidate validation"]["candidate_pair_recall"])
    recall_500 = float(results["500-candidate validation"]["candidate_pair_recall"])
    recall_250_minus_100 = recall_250 - recall_100
    recall_500_minus_250 = recall_500 - recall_250
    recall_500_minus_100 = recall_500 - recall_100
    pct_point_250_minus_100 = recall_250_minus_100 * 100.0
    pct_point_500_minus_250 = recall_500_minus_250 * 100.0
    pct_point_500_minus_100 = recall_500_minus_100 * 100.0

    metrics_to_show = [
        "validation_source1_entities",
        "total_candidate_pairs",
        "total_ground_truth_match_pairs",
        "true_match_pairs_captured",
        "candidate_pair_recall",
        "source1_entities_with_at_least_one_true_match_captured",
        "source1_entity_coverage_percentage",
        "average_candidates_per_source1_entity",
        "max_candidates_per_source1_entity",
        "source1_entities_hitting_cap",
        "source1_entities_hitting_cap_percentage",
    ]

    rows = []
    for metric in metrics_to_show:
        row = [metric]
        for label in ("100-candidate validation", "250-candidate validation", "500-candidate validation"):
            value = results[label][metric]
            if metric.endswith("percentage") or metric == "source1_entity_coverage_percentage" or metric == "source1_entities_hitting_cap_percentage":
                row.append(f"{float(value):.2f}%")
            elif metric in {"candidate_pair_recall"}:
                row.append(f"{float(value):.6f}")
            elif metric in {"average_candidates_per_source1_entity"}:
                row.append(f"{float(value):.2f}")
            else:
                row.append(str(int(value)))
        rows.append(row)

    print("Candidate-pair recall comparison for the validation split")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")
    print()
    print(format_table(["Metric", "100-candidate file", "250-candidate file", "500-candidate file"], rows))
    print()
    print("Candidate recall improvement")
    print(format_table(
        ["Metric", "Value"],
        [
            ["recall_250 - recall_100", f"{recall_250_minus_100:.6f}"],
            ["percentage-point improvement", f"{pct_point_250_minus_100:.2f} pp"],
            ["recall_500 - recall_250", f"{recall_500_minus_250:.6f}"],
            ["percentage-point improvement", f"{pct_point_500_minus_250:.2f} pp"],
            ["recall_500 - recall_100", f"{recall_500_minus_100:.6f}"],
            ["percentage-point improvement", f"{pct_point_500_minus_100:.2f} pp"],
        ],
    ))


if __name__ == "__main__":
    main()
