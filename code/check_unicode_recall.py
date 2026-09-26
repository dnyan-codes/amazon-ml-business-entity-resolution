import pandas as pd

GT_PATH = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv"

CANDIDATE_PATH = r"output\validation_test\unicode_training_candidates.tsv"

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    nrows=10000
)

print("Loading Unicode candidates...")

candidates = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

total_with_match = 0
found_match = 0

for _, row in candidates.iterrows():

    true_ids = {
        x.strip()
        for x in str(
            gt_map.get(
                row["source1_entity_id"],
                ""
            )
        ).split(",")
        if x.strip()
    }

    if not true_ids:
        continue

    total_with_match += 1

    candidate_ids = {
        x.strip()
        for x in str(
            row["candidate_entity_ids"]
        ).split(",")
        if x.strip()
    }

    if true_ids.intersection(candidate_ids):
        found_match += 1


print()
print("===================================")
print("UNICODE CANDIDATE RECALL")
print("===================================")

print(
    f"S1 rows with true match : "
    f"{total_with_match:,}"
)

print(
    f"S1 rows with candidate hit : "
    f"{found_match:,}"
)

if total_with_match:

    recall = found_match / total_with_match

    print(
        f"Candidate Recall : {recall:.4f}"
    )

    print(
        f"Candidate Recall : "
        f"{recall * 100:.2f}%"
    )