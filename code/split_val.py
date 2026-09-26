
import argparse
import csv
import os
import random
from collections import defaultdict


def load_tsv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def write_tsv_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Split train_source1 + ground_truth into train/val, stratified by country.")
    parser.add_argument("--source1", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    s1_fields, s1_rows = load_tsv_rows(args.source1)
    gt_fields, gt_rows = load_tsv_rows(args.ground_truth)

    gt_by_id = {row["source1_entity_id"]: row for row in gt_rows}

    # group entity_ids by country for stratified sampling
    by_country = defaultdict(list)
    for row in s1_rows:
        by_country[row["country"]].append(row["entity_id"])

    val_ids = set()
    for country, ids in by_country.items():
        ids = ids[:]  # copy
        rng.shuffle(ids)
        n_val = round(len(ids) * args.val_size)
        val_ids.update(ids[:n_val])

    train_s1_rows, val_s1_rows = [], []
    train_gt_rows, val_gt_rows = [], []
    missing_gt = 0

    for row in s1_rows:
        eid = row["entity_id"]
        gt_row = gt_by_id.get(eid)
        if gt_row is None:
            missing_gt += 1
            continue  # skip entities with no ground-truth row at all (shouldn't happen, but be safe)

        if eid in val_ids:
            val_s1_rows.append(row)
            val_gt_rows.append(gt_row)
        else:
            train_s1_rows.append(row)
            train_gt_rows.append(gt_row)

    if missing_gt:
        print(f"WARNING: {missing_gt} source1 entities had no ground-truth row and were dropped from the split")

    write_tsv_rows(os.path.join(args.out_dir, "train_source1.tsv"), s1_fields, train_s1_rows)
    write_tsv_rows(os.path.join(args.out_dir, "train_ground_truth.tsv"), gt_fields, train_gt_rows)
    write_tsv_rows(os.path.join(args.out_dir, "val_source1.tsv"), s1_fields, val_s1_rows)
    write_tsv_rows(os.path.join(args.out_dir, "val_ground_truth.tsv"), gt_fields, val_gt_rows)

    print(f"Train: {len(train_s1_rows)} entities")
    print(f"Val:   {len(val_s1_rows)} entities")

    for label, rows in [("Train", train_s1_rows), ("Val", val_s1_rows)]:
        counts = defaultdict(int)
        for row in rows:
            counts[row["country"]] += 1
        total = len(rows)
        breakdown = ", ".join(f"{c}={n} ({100*n/total:.1f}%)" for c, n in counts.items())
        print(f"{label} country breakdown: {breakdown}")

    print(f"\nFiles written to {args.out_dir}/")
    print("Reminder: Source 2 and Source 3 are NOT split — use the full "
          "train_source2.tsv / train_source3.tsv as candidate pools for both "
          "train and val runs.")


if __name__ == "__main__":
    main()