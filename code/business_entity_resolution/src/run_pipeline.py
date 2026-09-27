import os
import sys
import pandas as pd

# Add current src folder to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from preprocessing import load_source
from blocking import (
    generate_candidate_pairs,
    generate_candidate_pairs_for_matching
)
from matching import find_matches


# ============================================================
# PATHS
# ============================================================

# Your actual dataset location
DATASET_DIR = (
    r"C:\Users\Dnyaneshwari\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
    r"\dataset"
    r"\test"
)

# Your GitHub project output folder
PROJECT_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..")
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "output"
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DATASET FILE PATHS
# ============================================================

SOURCE1_PATH = os.path.join(
    DATASET_DIR,
    "test_source1.tsv"
)

SOURCE2_PATH = os.path.join(
    DATASET_DIR,
    "test_source2.tsv"
)

SOURCE3_PATH = os.path.join(
    DATASET_DIR,
    "test_source3.tsv"
)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking dataset files...")

for path in [
    SOURCE1_PATH,
    SOURCE2_PATH,
    SOURCE3_PATH
]:

    if not os.path.exists(path):

        print(
            f"\nERROR: File not found:\n{path}"
        )

        sys.exit(1)

    print(
        f"Found: {os.path.basename(path)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading test data...")

source1 = load_source(
    SOURCE1_PATH
)

source2 = load_source(
    SOURCE2_PATH
)

source3 = load_source(
    SOURCE3_PATH
)

print(
    f"Source 1 rows: {len(source1)}"
)

print(
    f"Source 2 rows: {len(source2)}"
)

print(
    f"Source 3 rows: {len(source3)}"
)


# ============================================================
# BLOCKING
# ============================================================

print("\nGenerating candidate pairs...")

candidate_output = generate_candidate_pairs(
    source1,
    source2,
    source3
)

candidate_pairs = generate_candidate_pairs_for_matching(
    source1,
    source2,
    source3
)

print(
    f"Candidate rows: {len(candidate_pairs)}"
)


# ============================================================
# SAVE CANDIDATE PAIRS
# ============================================================

candidate_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

candidate_output.to_csv(
    candidate_path,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print(
    f"\nCandidate file created:\n{candidate_path}"
)


# ============================================================
# MATCHING
# ============================================================

print("\nRunning entity matching...")

matching_results = find_matches(
    source1,
    source2,
    source3,
    candidate_pairs,
    threshold=0.70
)


# ============================================================
# MAKE SURE EVERY TEST S1 EXISTS
# ============================================================

required_ids = source1[
    "entity_id"
].tolist()

matching_results = (
    matching_results
    .set_index("source1_entity_id")
    .reindex(required_ids)
    .fillna("")
    .reset_index()
)


# ============================================================
# SAVE FINAL MATCHING RESULTS
# ============================================================

matching_path = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

matching_results.to_csv(
    matching_path,
    sep="\t",
    index=False,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

non_empty = (
    matching_results["matched_entity_ids"]
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

print(
    f"\nMatching file created:\n{matching_path}"
)

print(
    f"\nTotal Source-1 entities: "
    f"{len(matching_results)}"
)

print(
    f"Entities with at least one match: "
    f"{non_empty}"
)

print(
    f"Entities with no match: "
    f"{len(matching_results) - non_empty}"
)

print("\nPIPELINE COMPLETED.")