
import argparse
import os
import pandas as pd


def inspect_file(path, is_ground_truth=False):
    print("=" * 70)
    print(f"FILE: {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print("  !! FILE NOT FOUND — check the path / filename.")
        return

    df = pd.read_csv(path, sep="\t", dtype=str)  # dtype=str avoids silent numeric coercion

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nEmpty-string values per column (distinct from NaN):")
    for col in df.columns:
        empty_count = (df[col].fillna("").str.strip() == "").sum()
        print(f"  {col}: {empty_count}")

    id_col = "source1_entity_id" if is_ground_truth else "entity_id"
    if id_col in df.columns:
        dup_count = df[id_col].duplicated().sum()
        print(f"\nDuplicate {id_col} values: {dup_count}")
        if dup_count:
            print("  Examples:", df[df[id_col].duplicated()][id_col].head(5).tolist())

    if "country" in df.columns:
        print("\nCountry value counts:")
        print(df["country"].value_counts(dropna=False).to_string())

    if "business_name" in df.columns:
        print("\nSample business_name values:")
        print(df["business_name"].dropna().sample(min(5, len(df)), random_state=0).to_string(index=False))

    if "business_address" in df.columns:
        print("\nSample business_address values:")
        print(df["business_address"].dropna().sample(min(5, len(df)), random_state=0).to_string(index=False))

    if is_ground_truth and "matched_entity_ids" in df.columns:
        empty_matches = (df["matched_entity_ids"].fillna("").str.strip() == "").sum()
        print(f"\nSingletons (empty matched_entity_ids): {empty_matches} / {len(df)} "
              f"({100 * empty_matches / len(df):.1f}%)")
        match_counts = df["matched_entity_ids"].fillna("").apply(
            lambda s: 0 if s.strip() == "" else len(s.split(","))
        )
        print(f"Match-count distribution:\n{match_counts.value_counts().sort_index().to_string()}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Sanity-check the raw entity resolution TSVs.")
    parser.add_argument("--dataset-dir", required=True,
                         help="Path to dataset/train or dataset/test")
    args = parser.parse_args()

    prefix = "train" if "train" in os.path.basename(args.dataset_dir.rstrip("/")) else "test"

    for src in ("source1", "source2", "source3"):
        inspect_file(os.path.join(args.dataset_dir, f"{prefix}_{src}.tsv"))

    gt_path = os.path.join(args.dataset_dir, f"{prefix}_ground_truth.tsv")
    if os.path.exists(gt_path):
        inspect_file(gt_path, is_ground_truth=True)


if __name__ == "__main__":
    main()