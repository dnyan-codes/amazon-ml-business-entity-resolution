
import argparse
import csv
import sys
from collections import namedtuple

EntityScore = namedtuple(
    "EntityScore",
    ["entity_id", "precision", "recall", "f_half", "true_pos", "false_pos", "false_neg"],
)

BETA = 0.5
BETA_SQ = BETA * BETA


def parse_id_list(raw):
    """Turn a comma-separated id string (or empty/NaN) into a set of ids."""
    if raw is None:
        return set()
    raw = raw.strip()
    if raw == "" or raw.lower() == "nan":
        return set()
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def load_tsv(path, id_col, list_col):
    """Load a two-column TSV into {source1_entity_id: set(other_ids)}."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if id_col not in reader.fieldnames or list_col not in reader.fieldnames:
            raise ValueError(
                f"{path}: expected columns '{id_col}' and '{list_col}', "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            eid = row[id_col].strip()
            if eid in data:
                print(f"WARNING: duplicate {id_col} '{eid}' in {path} — "
                      f"keeping first occurrence", file=sys.stderr)
                continue
            data[eid] = parse_id_list(row[list_col])
    return data


def score_entity(entity_id, true_ids, pred_ids):
    """Precision / Recall / F_0.5 for a single Source-1 entity."""
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    if not true_ids and not pred_ids:
        # Correctly predicted empty (singleton) -> full credit.
        precision, recall, f_half = 1.0, 1.0, 1.0
    else:
        precision = tp / len(pred_ids) if pred_ids else 0.0
        recall = tp / len(true_ids) if true_ids else 0.0
        if precision == 0.0 and recall == 0.0:
            f_half = 0.0
        else:
            f_half = ((1 + BETA_SQ) * precision * recall) / (BETA_SQ * precision + recall)

    return EntityScore(entity_id, precision, recall, f_half, tp, fp, fn)


def evaluate(ground_truth, predictions):
    """
    ground_truth / predictions: {source1_entity_id: set(matched_ids)}
    Returns (macro_f_half, list[EntityScore]) over the union of entity ids
    present in ground_truth (missing predictions are treated as empty lists).
    """
    scores = []
    missing_preds = []
    for eid, true_ids in ground_truth.items():
        pred_ids = predictions.get(eid)
        if pred_ids is None:
            missing_preds.append(eid)
            pred_ids = set()
        scores.append(score_entity(eid, true_ids, pred_ids))

    if missing_preds:
        print(f"WARNING: {len(missing_preds)} ground-truth entities had no row "
              f"in predictions — scored as empty predictions "
              f"(e.g. {missing_preds[:5]})", file=sys.stderr)

    extra_preds = set(predictions) - set(ground_truth)
    if extra_preds:
        print(f"NOTE: {len(extra_preds)} predicted entity ids are not in the "
              f"ground-truth file and were ignored for scoring "
              f"(e.g. {sorted(extra_preds)[:5]})", file=sys.stderr)

    macro_f_half = sum(s.f_half for s in scores) / len(scores) if scores else 0.0
    return macro_f_half, scores


def categorize_examples(scores, ground_truth, predictions, n):
    """Pick up to n example rows for each of the four illustrative categories."""
    true_match, false_match, missed_match, correct_empty = [], [], [], []

    for s in scores:
        true_ids = ground_truth.get(s.entity_id, set())
        pred_ids = predictions.get(s.entity_id, set())

        if not true_ids and not pred_ids:
            correct_empty.append(s)
            continue
        if s.true_pos > 0:
            true_match.append(s)
        if s.false_pos > 0:
            false_match.append(s)
        if s.false_neg > 0:
            missed_match.append(s)

    def fmt(s):
        true_ids = ground_truth.get(s.entity_id, set())
        pred_ids = predictions.get(s.entity_id, set())
        return (f"  {s.entity_id}: true={sorted(true_ids)} pred={sorted(pred_ids)} "
                f"(P={s.precision:.2f} R={s.recall:.2f} F0.5={s.f_half:.2f})")

    print(f"\n--- True matches (correct hits) [{len(true_match)} total] ---")
    for s in true_match[:n]:
        print(fmt(s))

    print(f"\n--- False matches (wrong merges — hurts precision) [{len(false_match)} total] ---")
    for s in false_match[:n]:
        print(fmt(s))

    print(f"\n--- Missed matches (false negatives) [{len(missed_match)} total] ---")
    for s in missed_match[:n]:
        print(fmt(s))

    print(f"\n--- Correct empty predictions (singletons) [{len(correct_empty)} total] ---")
    for s in correct_empty[:n]:
        print(fmt(s))


def main():
    parser = argparse.ArgumentParser(description="Score matching_results.tsv against ground truth (macro F_0.5).")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth TSV "
                         "(source1_entity_id, matched_entity_ids)")
    parser.add_argument("--predictions", required=True, help="Path to predictions TSV, "
                         "same columns as matching_results.tsv")
    parser.add_argument("--examples", type=int, default=5, help="Examples to print per category (default 5)")
    parser.add_argument("--id-col", default="source1_entity_id")
    parser.add_argument("--gt-list-col", default="matched_entity_ids")
    parser.add_argument("--pred-list-col", default="matched_entity_ids")
    args = parser.parse_args()

    ground_truth = load_tsv(args.ground_truth, args.id_col, args.gt_list_col)
    predictions = load_tsv(args.predictions, args.id_col, args.pred_list_col)

    macro_f_half, scores = evaluate(ground_truth, predictions)

    precisions = [s.precision for s in scores]
    recalls = [s.recall for s in scores]
    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0

    total_tp = sum(s.true_pos for s in scores)
    total_fp = sum(s.false_pos for s in scores)
    total_fn = sum(s.false_neg for s in scores)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f_half = (
        ((1 + BETA_SQ) * micro_p * micro_r) / (BETA_SQ * micro_p + micro_r)
        if (micro_p or micro_r) else 0.0
    )

    print("=" * 60)
    print(f"Entities scored:        {len(scores)}")
    print(f"Macro-averaged F_0.5:   {macro_f_half:.4f}   <-- this is the leaderboard metric")
    print(f"Macro-averaged Precision: {macro_p:.4f}")
    print(f"Macro-averaged Recall:    {macro_r:.4f}")
    print("-" * 60)
    print(f"(diagnostic) Micro Precision: {micro_p:.4f}  Recall: {micro_r:.4f}  F0.5: {micro_f_half:.4f}")
    print(f"(diagnostic) Total TP={total_tp} FP={total_fp} FN={total_fn}")
    print("=" * 60)

    if args.examples > 0:
        categorize_examples(scores, ground_truth, predictions, args.examples)


if __name__ == "__main__":
    main()