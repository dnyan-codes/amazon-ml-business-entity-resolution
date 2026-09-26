import os
import sys


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    CURRENT_DIR
)

from preprocessing import load_source
from blocking import generate_candidate_pairs


DATASET_DIR = (
    r"C:\Users\Dnyaneshwari\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
    r"\dataset"
    r"\test"
)


PROJECT_DIR = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "..",
        ".."
    )
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "output"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("Loading Source 1...")

source1 = load_source(
    os.path.join(
        DATASET_DIR,
        "test_source1.tsv"
    )
)

print(
    f"Source 1 rows: {len(source1):,}"
)


print("Loading Source 2...")

source2 = load_source(
    os.path.join(
        DATASET_DIR,
        "test_source2.tsv"
    )
)

print(
    f"Source 2 rows: {len(source2):,}"
)


print("Loading Source 3...")

source3 = load_source(
    os.path.join(
        DATASET_DIR,
        "test_source3.tsv"
    )
)

print(
    f"Source 3 rows: {len(source3):,}"
)


output_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


print()
print("Generating candidate pairs...")


generate_candidate_pairs(
    source1,
    source2,
    source3,
    output_path,
    max_candidates=100
)


print()
print("DONE!")
print(output_path)