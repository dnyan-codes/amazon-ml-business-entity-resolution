import csv
import os
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = (
    r"C:\Users\Dnyaneshwari\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
    r"\dataset"
    r"\test"
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "output"
)

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

MATCHING_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.replace(
        "&",
        " and "
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SIMILARITY
# ============================================================

def similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("Loading Source 1...")

source1 = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "test_source1.tsv"
    ),
    sep="\t"
)

print(
    f"Source 1 rows: {len(source1):,}"
)


print("Loading Source 2...")

source2 = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "test_source2.tsv"
    ),
    sep="\t"
)

print(
    f"Source 2 rows: {len(source2):,}"
)


print("Loading Source 3...")

source3 = pd.read_csv(
    os.path.join(
        DATASET_DIR,
        "test_source3.tsv"
    ),
    sep="\t"
)

print(
    f"Source 3 rows: {len(source3):,}"
)


# ============================================================
# CREATE COMPACT LOOKUPS
# ============================================================

print("Preparing entity lookups...")


def prepare_lookup(df):

    lookup = {}

    for row in df.itertuples(index=False):

        lookup[row.entity_id] = (
            normalize_text(row.business_name),
            normalize_text(row.business_address),
            normalize_text(row.country)
        )

    return lookup


source2_lookup = prepare_lookup(
    source2
)

source3_lookup = prepare_lookup(
    source3
)


# Source 1 lookup

source1_lookup = {}

for row in source1.itertuples(index=False):

    source1_lookup[row.entity_id] = (
        normalize_text(row.business_name),
        normalize_text(row.business_address),
        normalize_text(row.country)
    )


# ============================================================
# MATCH SCORE
# ============================================================

def calculate_score(
    source1_data,
    candidate_data
):

    s1_name, s1_address, s1_country = source1_data

    c_name, c_address, c_country = candidate_data

    name_score = similarity(
        s1_name,
        c_name
    )

    address_score = similarity(
        s1_address,
        c_address
    )

    country_score = (
        1.0
        if (
            s1_country
            and c_country
            and s1_country == c_country
        )
        else 0.0
    )

    # Name is the strongest signal.
    score = (
        0.60 * name_score
        + 0.30 * address_score
        + 0.10 * country_score
    )

    return score


# ============================================================
# PROCESS CANDIDATE FILE
# ============================================================

print()
print("Processing candidate_pairs.tsv...")
print(
    "This will take some time because the file is ~1.94 GB."
)
print()


with open(
    CANDIDATE_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as candidate_file, open(
    MATCHING_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as output_file:

    reader = csv.DictReader(
        candidate_file,
        delimiter="\t"
    )

    writer = csv.writer(
        output_file,
        delimiter="\t"
    )

    writer.writerow(
        [
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

    processed = 0

    for row in reader:

        s1_id = row[
            "source1_entity_id"
        ]

        candidate_ids_text = row.get(
            "candidate_entity_ids",
            ""
        )

        candidate_ids = [
            x.strip()
            for x in candidate_ids_text.split(",")
            if x.strip()
        ]

        s1_data = source1_lookup.get(
            s1_id
        )

        if s1_data is None:

            writer.writerow(
                [
                    s1_id,
                    ""
                ]
            )

            continue

        scored_matches = []

        for candidate_id in candidate_ids:

            if candidate_id.startswith(
                "S2-"
            ):

                candidate_data = (
                    source2_lookup.get(
                        candidate_id
                    )
                )

            elif candidate_id.startswith(
                "S3-"
            ):

                candidate_data = (
                    source3_lookup.get(
                        candidate_id
                    )
                )

            else:

                continue

            if candidate_data is None:
                continue

            score = calculate_score(
                s1_data,
                candidate_data
            )

            # Matching threshold
            if score >= 0.70:

                scored_matches.append(
                    (
                        candidate_id,
                        score
                    )
                )

        # Highest score first
        scored_matches.sort(
            key=lambda x: x[1],
            reverse=True
        )

        matched_ids = [
            item[0]
            for item in scored_matches
        ]

        writer.writerow(
            [
                s1_id,
                ",".join(matched_ids)
            ]
        )

        processed += 1

        if processed % 10000 == 0:

            print(
                f"Processed "
                f"{processed:,} / "
                f"{len(source1):,}"
            )


print()
print("==========================================")
print("MATCHING COMPLETED")
print("==========================================")
print(
    f"Output: {MATCHING_FILE}"
)
print(
    f"Rows: {len(source1):,}"
)
print("==========================================")