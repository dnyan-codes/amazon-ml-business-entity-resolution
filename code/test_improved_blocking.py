import pandas as pd
import os
import csv
import re

BASE = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

S1_PATH = os.path.join(BASE, "train_source1.tsv")
S2_PATH = os.path.join(BASE, "train_source2.tsv")
S3_PATH = os.path.join(BASE, "train_source3.tsv")

OUTPUT = r"output\validation_test\improved_training_candidates.tsv"


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].map(normalize)
    df["address_norm"] = df["business_address"].map(normalize)
    df["country_norm"] = df["country"].fillna("").astype(str).str.lower().str.strip()

    df["name_prefix"] = (
        df["country_norm"] + "|" +
        df["name_norm"].str[:6]
    )

    df["name_suffix"] = (
        df["country_norm"] + "|" +
        df["name_norm"].str[-6:]
    )

    df["address_prefix"] = (
        df["country_norm"] + "|" +
        df["address_norm"].str[:8]
    )

    df["address_suffix"] = (
        df["country_norm"] + "|" +
        df["address_norm"].str[-8:]
    )

    # First character of first two name words
    def token_key(x):
        tokens = x.split()

        if len(tokens) >= 2:
            return tokens[0][:3] + "|" + tokens[1][:3]

        if len(tokens) == 1:
            return tokens[0][:6]

        return ""

    df["name_token_key"] = (
        df["country_norm"] + "|" +
        df["name_norm"].map(token_key)
    )

    return df


def build_lookup(df, column):

    lookup = {}

    for key, group in df.groupby(column):

        if not key or key.endswith("|"):
            continue

        lookup[key] = group["entity_id"].tolist()

    return lookup


print("Loading 10,000 Source-1 rows...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=10000
)

print("Loading Source-2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str
)

print("Loading Source-3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str
)

print("Preparing data...")

s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)

keys = [
    "name_prefix",
    "name_suffix",
    "address_prefix",
    "address_suffix",
    "name_token_key"
]

print("Building lookups...")

lookups = {}

for key in keys:

    print("  ", key)

    lookups[("s2", key)] = build_lookup(s2, key)
    lookups[("s3", key)] = build_lookup(s3, key)


print("Generating improved candidates...")

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(
        file,
        delimiter="\t"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    for index, row in enumerate(
        s1.iterrows(),
        start=1
    ):

        _, r = row

        candidates = []

        for key in keys:

            value = r[key]

            candidates.extend(
                lookups[("s2", key)].get(value, [])
            )

            candidates.extend(
                lookups[("s3", key)].get(value, [])
            )

        # Remove duplicates
        candidates = list(
            dict.fromkeys(candidates)
        )

        # Keep maximum 200
        candidates = candidates[:200]

        writer.writerow([
            r["entity_id"],
            ",".join(map(str, candidates))
        ])

        if index % 1000 == 0:
            print(
                f"Processed {index:,} / 10,000"
            )

print()
print("===================================")
print("IMPROVED BLOCKING COMPLETED")
print("===================================")
print("Output:", OUTPUT)