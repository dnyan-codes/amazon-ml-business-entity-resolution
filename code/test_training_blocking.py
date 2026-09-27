import pandas as pd
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "business_entity_resolution",
            "src"
        )
    )
)

from preprocessing import preprocess_dataframe
from blocking import generate_candidate_pairs


BASE = r"C:\Users\Dnyaneshwari\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

S1_PATH = os.path.join(BASE, "train_source1.tsv")
S2_PATH = os.path.join(BASE, "train_source2.tsv")
S3_PATH = os.path.join(BASE, "train_source3.tsv")

OUTPUT = r"output\validation_test\training_candidates.tsv"


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


print()
print("Normalizing Source-1...")
s1 = preprocess_dataframe(s1)

print("Normalizing Source-2...")
s2 = preprocess_dataframe(s2)

print("Normalizing Source-3...")
s3 = preprocess_dataframe(s3)


print()
print("Starting blocking...")

generate_candidate_pairs(
    s1,
    s2,
    s3,
    OUTPUT,
    max_candidates=100
)


print()
print("===================================")
print("TRAINING BLOCKING TEST COMPLETED")
print("===================================")
print("Output:", OUTPUT)