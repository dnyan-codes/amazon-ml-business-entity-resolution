import pandas as pd
import re
import unicodedata
from collections import defaultdict


BASE = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource"

S1_PATH = BASE + r"\dataset\train\train_source1.tsv"
S2_PATH = BASE + r"\dataset\train\train_source2.tsv"
S3_PATH = BASE + r"\dataset\train\train_source3.tsv"
GT_PATH = BASE + r"\dataset\train\train_ground_truth.tsv"

OUTPUT = r"output\validation_test\token_training_candidates.tsv"


def normalize(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("&", " and ")

    # Keep Unicode letters/numbers
    text = "".join(
        ch if (ch.isalnum() or ch.isspace()) else " "
        for ch in text
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_tokens(text):
    text = normalize(text)

    tokens = text.split()

    # Remove very short/common noise tokens
    tokens = [
        t for t in tokens
        if len(t) >= 3
    ]

    return tokens


print("Loading 10,000 Source-1 rows...")
s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=10000
).fillna("")

print("Loading Source-2...")
s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str
).fillna("")

print("Loading Source-3...")
s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str
).fillna("")


print("Preparing Source-2 and Source-3 token indexes...")

# country + token prefix
name_index = defaultdict(set)
address_index = defaultdict(set)

for df in [s2, s3]:

    for _, row in df.iterrows():

        entity_id = row["entity_id"]
        country = normalize(row["country"])

        # BUSINESS NAME TOKENS
        name_tokens = get_tokens(row["business_name"])

        # Use token prefixes to tolerate small spelling differences
        for token in name_tokens:

            key = country + "|" + token[:4]

            name_index[key].add(entity_id)

        # ADDRESS TOKENS
        address_tokens = get_tokens(row["business_address"])

        for token in address_tokens:

            key = country + "|" + token[:5]

            address_index[key].add(entity_id)


print("Generating candidates...")

rows = []

for i, (_, row) in enumerate(s1.iterrows(), start=1):

    s1_id = row["entity_id"]
    country = normalize(row["country"])

    candidates = set()

    # NAME TOKEN BLOCKING
    name_tokens = get_tokens(row["business_name"])

    for token in name_tokens:

        key = country + "|" + token[:4]

        if key in name_index:
            candidates.update(name_index[key])

    # ADDRESS TOKEN BLOCKING
    address_tokens = get_tokens(row["business_address"])

    for token in address_tokens:

        key = country + "|" + token[:5]

        if key in address_index:
            candidates.update(address_index[key])

    # Keep candidate count manageable
    candidates = list(candidates)[:300]

    rows.append({
        "source1_entity_id": s1_id,
        "candidate_entity_ids": ",".join(candidates)
    })

    if i % 1000 == 0:
        print(f"Processed {i:,} / 10,000")


result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print()
print("===================================")
print("TOKEN BLOCKING COMPLETED")
print("===================================")
print("Output:", OUTPUT)
print("Rows:", len(result))