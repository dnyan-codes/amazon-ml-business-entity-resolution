# amazon-ml-2026 — Business Entity Resolution

Team repo for the ML Challenge 2026 Business Entity Resolution task. Pipeline:

```
Source 1 + Source 2 + Source 3
        │
        ▼
1. CLEAN TEXT           code/normalize.py      (Member 1)
        │
        ▼
2. CANDIDATE GENERATION  code/blocking.py       (Member 1)
        │
        ▼
3. SIMILARITY FEATURES   code/features.py       (Member 2)
        │
        ▼
4. ML MODEL (match/no)   code/model.py          (Member 2)
        │
        ▼
5. EVALUATION            code/evaluate.py       (Member 3)
        │
        ▼
output/matching_results.tsv
output/candidate_pairs.tsv
```

## Repo layout

```
amazon-ml-2026/
├── student_resource/            # challenge-provided data + official validator
│   ├── dataset/train/
│   ├── dataset/test/
│   └── utils/validate_submission.py
├── code/
│   ├── normalize.py             # Member 1 — text cleaning
│   ├── blocking.py              # Member 1 — candidate generation
│   ├── features.py              # Member 2 — similarity features
│   ├── model.py                 # Member 2 — classifier train/predict
│   └── evaluate.py              # Member 3 — macro F_0.5 scoring
├── output/
│   ├── matching_results.tsv     # final matches (leaderboard file)
│   └── candidate_pairs.tsv      # blocking candidates (audit file)
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pin exact versions in `requirements.txt` once the model choice (e.g. LightGBM/XGBoost)
is locked in — required for the final submission package.

## Running the pipeline end-to-end

```bash
# 1. Clean
python3 code/normalize.py --in student_resource/dataset/train --out data/clean

# 2. Block
python3 code/blocking.py --in data/clean --out output/candidate_pairs.tsv

# 3. Features + model
python3 code/features.py --candidates output/candidate_pairs.tsv --out data/features.csv
python3 code/model.py --features data/features.csv --out output/matching_results.tsv

# 4. Score against a held-out validation split
python3 code/evaluate.py \
    --ground-truth data/val_ground_truth.tsv \
    --predictions output/matching_results.tsv \
    --examples 5

# 5. Validate submission format before uploading
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

(Exact CLI flags above are placeholders — align them with however Member 1/2 actually
build `normalize.py` / `blocking.py` / `features.py` / `model.py`.)

## Branching

Each member works on their own branch (`member1-blocking`, `member2-model`,
`member3-eval`) and opens a PR into `main`. Keep file boundaries as listed above so
merges stay conflict-free.

## Scoring metric

Macro-averaged F_0.5 per Source-1 entity (precision weighted 2x over recall).
Singletons (no true match) score 1.0 for a correct empty prediction, 0.0 for any
false match. See `code/evaluate.py` for the local scoring implementation and
`student_resource/README.md` for the official formula and leaderboard rules.

## Fair play

No external APIs / databases / geocoding lookups for entity resolution — training
data only. Final model must be MIT/Apache-2.0 licensed and ≤8B parameters.