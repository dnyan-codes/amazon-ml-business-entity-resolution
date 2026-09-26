import pandas as pd

GT_PATH = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv"
CANDIDATE_PATH = r"output\candidate_pairs.tsv"

SAMPLE_SIZE = 10000

print("Loading ground truth sample...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    nrows=SAMPLE_SIZE,
    dtype=str
)

print("Loading candidate sample...")

candidates = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    nrows=SAMPLE_SIZE,
    dtype=str
)

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

found = 0
has_match = 0

for _, row in candidates.iterrows():

    s1_id = row["source1_entity_id"]

    true_ids = set(
        x.strip()
        for x in str(gt_map.get(s1_id, "")).split(",")
        if x.strip()
    )

    if not true_ids:
        continue

    has_match += 1

    candidate_ids = set(
        x.strip()
        for x in str(row["candidate_entity_ids"]).split(",")
        if x.strip()
    )

    if true_ids & candidate_ids:
        found += 1

print()
print("===================================")
print("Candidate Recall Check")
print("===================================")
print(f"Sample S1 rows with true matches : {has_match:,}")
print(f"True matches found in candidates : {found:,}")

if has_match:
    recall = found / has_match
    print(f"Candidate recall                 : {recall:.4f}")
    print(f"Candidate recall (%)             : {recall * 100:.2f}%")
else:
    print("No matched entities found in sample.")