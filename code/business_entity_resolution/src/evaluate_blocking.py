import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = (
    r"C:\Users\Dnyaneshwari\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
    r"\dataset"
    r"\train"
)


# ============================================================
# BLOCKING KEY FUNCTION
# ============================================================

def prepare_keys(df):

    name = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    address = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    country = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return (
        country + "|" + name.str[:6],
        country + "|" + address.str[:8]
    )


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "train_source1.tsv"
    ),
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print(f"Source 1: {len(s1):,}")


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("Loading Source 2...")

s2 = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "train_source2.tsv"
    ),
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print(f"Source 2: {len(s2):,}")


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("Loading Source 3...")

s3 = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "train_source3.tsv"
    ),
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print(f"Source 3: {len(s3):,}")


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

print("Creating blocking keys...")

s1["name_key"], s1["address_key"] = prepare_keys(s1)
s2["name_key"], s2["address_key"] = prepare_keys(s2)
s3["name_key"], s3["address_key"] = prepare_keys(s3)


# ============================================================
# CREATE SOURCE ID + KEY TABLE
# ============================================================

print("Creating lookup tables...")

s2_keys = pd.concat(
    [
        s2[["entity_id", "name_key"]],
        s2[["entity_id", "address_key"]]
        .rename(columns={"address_key": "name_key"})
    ],
    ignore_index=True
)

s3_keys = pd.concat(
    [
        s3[["entity_id", "name_key"]],
        s3[["entity_id", "address_key"]]
        .rename(columns={"address_key": "name_key"})
    ],
    ignore_index=True
)

s2_keys = s2_keys.rename(
    columns={"name_key": "block_key"}
)

s3_keys = s3_keys.rename(
    columns={"name_key": "block_key"}
)


# ============================================================
# GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "train_ground_truth.tsv"
    ),
    sep="\t"
)

print(f"Ground-truth rows: {len(gt):,}")


# ============================================================
# SOURCE 1 KEYS
# ============================================================

s1_keys = s1[
    [
        "entity_id",
        "name_key",
        "address_key"
    ]
].copy()

s1_keys = s1_keys.rename(
    columns={
        "entity_id": "source1_entity_id"
    }
)


# ============================================================
# MERGE GROUND TRUTH WITH S1 KEYS
# ============================================================

print("Preparing ground truth...")

gt = gt.merge(
    s1_keys,
    on="source1_entity_id",
    how="left"
)


# ============================================================
# BUILD BLOCK CANDIDATES
# ============================================================

print("Checking candidate blocks...")


# Convert ground truth matched IDs into sets
gt["true_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
    .str.split(",")
    .apply(
        lambda x: set(
            item.strip()
            for item in x
            if item.strip()
        )
    )
)


# ============================================================
# LOOKUP DICTIONARIES
# ============================================================

print("Building compact dictionaries...")

s2_name_lookup = (
    s2.groupby("name_key")["entity_id"]
    .agg(set)
    .to_dict()
)

s2_address_lookup = (
    s2.groupby("address_key")["entity_id"]
    .agg(set)
    .to_dict()
)

s3_name_lookup = (
    s3.groupby("name_key")["entity_id"]
    .agg(set)
    .to_dict()
)

s3_address_lookup = (
    s3.groupby("address_key")["entity_id"]
    .agg(set)
    .to_dict()
)


# ============================================================
# FAST EVALUATION
# ============================================================

print("Evaluating recall...")

total_true = 0
found_true = 0

all_found = 0
some_found = 0
none_found = 0


for row in gt.itertuples(index=False):

    true_ids = row.true_ids

    total_true += len(true_ids)

    candidates = set()

    candidates.update(
        s2_name_lookup.get(
            row.name_key,
            set()
        )
    )

    candidates.update(
        s3_name_lookup.get(
            row.name_key,
            set()
        )
    )

    candidates.update(
        s2_address_lookup.get(
            row.address_key,
            set()
        )
    )

    candidates.update(
        s3_address_lookup.get(
            row.address_key,
            set()
        )
    )

    found = true_ids.intersection(
        candidates
    )

    found_true += len(found)

    if len(found) == len(true_ids):
        all_found += 1

    elif len(found) > 0:
        some_found += 1

    else:
        none_found += 1


# ============================================================
# FINAL RESULTS
# ============================================================

recall = (
    found_true / total_true
    if total_true > 0
    else 0
)

print()
print("==========================================")
print("BLOCKING RECALL RESULTS")
print("==========================================")

print(
    f"Total true matches : {total_true:,}"
)

print(
    f"True matches found : {found_true:,}"
)

print(
    f"Candidate recall   : {recall * 100:.2f}%"
)

print()

print(
    f"S1 with ALL matches found  : {all_found:,}"
)

print(
    f"S1 with SOME matches found : {some_found:,}"
)

print(
    f"S1 with NO matches found   : {none_found:,}"
)

print("==========================================")