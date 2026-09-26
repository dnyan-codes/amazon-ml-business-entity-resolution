import pandas as pd
import os

BASE = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

GT_PATH = os.path.join(BASE, "train_ground_truth.tsv")
S1_PATH = os.path.join(BASE, "train_source1.tsv")
S2_PATH = os.path.join(BASE, "train_source2.tsv")
S3_PATH = os.path.join(BASE, "train_source3.tsv")

CANDIDATE_PATH = r"output\validation_test\improved_training_candidates.tsv"

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    nrows=10000
)

print("Loading candidates...")

candidates = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str
)

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=10000
)

print("Loading Source 2 IDs...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Loading Source 3 IDs...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1_map = s1.set_index("entity_id").to_dict("index")
s2_map = s2.set_index("entity_id").to_dict("index")
s3_map = s3.set_index("entity_id").to_dict("index")

candidate_map = dict(
    zip(
        candidates["source1_entity_id"],
        candidates["candidate_entity_ids"].fillna("")
    )
)

shown = 0

print()
print("===================================")
print("MISSED TRUE MATCHES")
print("===================================")

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    true_ids = {
        x.strip()
        for x in str(row["matched_entity_ids"]).split(",")
        if x.strip()
    }

    if not true_ids:
        continue

    candidate_ids = {
        x.strip()
        for x in str(
            candidate_map.get(s1_id, "")
        ).split(",")
        if x.strip()
    }

    missed = true_ids - candidate_ids

    if not missed:
        continue

    print()
    print("-----------------------------------")
    print("SOURCE 1")
    print("ID:", s1_id)

    s1_row = s1_map.get(s1_id)

    if s1_row:
        print("Name   :", s1_row["business_name"])
        print("Address:", s1_row["business_address"])
        print("Country:", s1_row["country"])

    for true_id in list(missed)[:2]:

        print()
        print("TRUE MATCH:", true_id)

        match = s2_map.get(true_id)

        if match is None:
            match = s3_map.get(true_id)

        if match:
            print("Name   :", match["business_name"])
            print("Address:", match["business_address"])
            print("Country:", match["country"])
        else:
            print("Could not locate entity.")

    shown += 1

    if shown >= 10:
        break

print()
print("===================================")
print("Displayed missed cases:", shown)
print("===================================")