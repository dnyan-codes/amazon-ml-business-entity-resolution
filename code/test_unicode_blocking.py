import pandas as pd
import os
import csv
import re
import unicodedata

BASE = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

S1_PATH = os.path.join(BASE, "train_source1.tsv")
S2_PATH = os.path.join(BASE, "train_source2.tsv")
S3_PATH = os.path.join(BASE, "train_source3.tsv")

OUTPUT = r"output\validation_test\unicode_training_candidates.tsv"


def normalize_unicode(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("&", " and ")

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Keep Unicode letters/numbers.
    # Remove punctuation.
    text = "".join(
        ch if (
            ch.isalnum() or ch.isspace()
        ) else " "
        for ch in text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def transliteration_key(text):

    """
    Create an ASCII representation for
    languages where possible.

    Characters that cannot be transliterated
    are simply ignored.
    """

    if not text:
        return ""

    normalized = unicodedata.normalize(
        "NFKD",
        text
    )

    ascii_text = normalized.encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    ascii_text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        ascii_text.lower()
    )

    ascii_text = re.sub(
        r"\s+",
        " ",
        ascii_text
    ).strip()

    return ascii_text


def prepare(df):

    df = df.copy()

    df["name_unicode"] = (
        df["business_name"]
        .map(normalize_unicode)
    )

    df["address_unicode"] = (
        df["business_address"]
        .map(normalize_unicode)
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["name_ascii"] = (
        df["name_unicode"]
        .map(transliteration_key)
    )

    df["address_ascii"] = (
        df["address_unicode"]
        .map(transliteration_key)
    )

    return df


def build_lookup(df, column):

    lookup = {}

    for key, group in df.groupby(column):

        if not key:
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

print("Building lookups...")

lookups = {}

for source_name, df in [
    ("s2", s2),
    ("s3", s3)
]:

    lookups[(source_name, "name_unicode")] = \
        build_lookup(df, "name_unicode")

    lookups[(source_name, "name_ascii")] = \
        build_lookup(df, "name_ascii")

    lookups[(source_name, "address_unicode")] = \
        build_lookup(df, "address_unicode")

    lookups[(source_name, "address_ascii")] = \
        build_lookup(df, "address_ascii")


print("Generating candidates...")

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

    for index, (_, row) in enumerate(
        s1.iterrows(),
        start=1
    ):

        candidates = []

        for key in [
            "name_unicode",
            "name_ascii",
            "address_unicode",
            "address_ascii"
        ]:

            value = row[key]

            candidates.extend(
                lookups[("s2", key)].get(
                    value,
                    []
                )
            )

            candidates.extend(
                lookups[("s3", key)].get(
                    value,
                    []
                )
            )

        candidates = list(
            dict.fromkeys(candidates)
        )

        candidates = candidates[:200]

        writer.writerow([
            row["entity_id"],
            ",".join(map(str, candidates))
        ])

        if index % 1000 == 0:
            print(
                f"Processed {index:,} / 10,000"
            )

print()
print("===================================")
print("UNICODE BLOCKING COMPLETED")
print("===================================")
print("Output:", OUTPUT)