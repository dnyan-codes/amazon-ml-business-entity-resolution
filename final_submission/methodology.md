# Amazon ML Business Entity Resolution — Methodology

## 1. Problem Overview

The objective is to identify which Source 2 and Source 3 business entities correspond to each Source 1 business entity. The input sources contain business name, business address, country, and entity identifiers. The task is challenging because the same business can appear with different spellings, formatting, addresses, and representations across sources.

## 2. Data Preprocessing

Business names and addresses are normalized before matching.

The preprocessing steps include:
- Converting text to lowercase.
- Normalizing Unicode text.
- Replacing ampersand with "and".
- Removing punctuation and special characters.
- Removing extra whitespace.
- Normalizing country values.

## 3. Candidate Generation / Blocking

Comparing every Source 1 entity with every Source 2 and Source 3 entity would require an extremely large number of comparisons.

A blocking strategy is therefore used to generate a smaller candidate set.

Two blocking keys are created:
- Country + first six characters of the normalized business name.
- Country + first eight characters of the normalized business address.

Candidates from both blocking strategies are combined, duplicate IDs are removed, and a maximum of 100 candidates is retained per Source 1 entity.

The generated candidate pairs are stored in output/candidate_pairs.tsv.

## 4. Matching and Feature Engineering

Candidate pairs are compared using business name, business address, and country information.

String similarity is calculated for business names and addresses. Country agreement is also used as a matching feature.

The project also contains a Logistic Regression based entity matching pipeline using engineered pairwise features.

## 5. Final Matching

The generated final prediction file uses a combined matching score based on:
- Business name similarity: 60%
- Business address similarity: 30%
- Country agreement: 10%

Candidate pairs meeting the matching threshold are included in the final prediction.

The final predictions are stored in output/matching_results.tsv.

## 6. Validation

The official challenge validator was used to check the submission format and entity coverage.

The final matching file contains exactly 1,732,544 Source 1 entities, with one row for every required Source 1 entity.

The required output columns are:
- source1_entity_id
- matched_entity_ids

## 7. Project Structure

The runnable entity-resolution implementation is provided under:

code/business_entity_resolution/

The project also contains feature engineering, model, evaluation, sanity-check, smoke-test, and validation components.

## 8. Final Submission Files

The final package contains:
- output/matching_results.tsv
- output/candidate_pairs.tsv
- code/business_entity_resolution/
- methodology.md
