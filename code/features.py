from __future__ import annotations

import csv
import re
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from rapidfuzz import fuzz


def normalize_text(value: object) -> str:
    """Deterministic, feature-level normalization for comparison only.

    This is intentionally lightweight and does not replace Member 1's full
    normalization/blocking pipeline. It is only used when comparing strings from
    candidate pairs and labels.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_tokens(value: object) -> List[str]:
    normalized = normalize_text(value)
    return [tok for tok in normalized.split() if tok]


def _exact_match(a: object, b: object) -> int:
    return 1 if normalize_text(a) == normalize_text(b) and normalize_text(a) != "" else 0


def _token_overlap(a: object, b: object) -> float:
    tokens_a = set(_safe_tokens(a))
    tokens_b = set(_safe_tokens(b))
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / max(len(tokens_a | tokens_b), 1)


def _token_count(a: object) -> int:
    return len(_safe_tokens(a))


def _length_difference(a: object, b: object) -> int:
    return abs(len(normalize_text(a)) - len(normalize_text(b)))


def generate_pair_features(source1_record: Mapping[str, object], candidate_record: Mapping[str, object]) -> Dict[str, float | int]:
    """Generate pairwise similarity features for one source1/candidate pair."""
    s1_name = source1_record.get("business_name", "")
    cand_name = candidate_record.get("business_name", "")
    s1_address = source1_record.get("business_address", "")
    cand_address = candidate_record.get("business_address", "")
    s1_country = source1_record.get("country", "")
    cand_country = candidate_record.get("country", "")

    s1_name_norm = normalize_text(s1_name)
    cand_name_norm = normalize_text(cand_name)
    s1_addr_norm = normalize_text(s1_address)
    cand_addr_norm = normalize_text(cand_address)

    features: Dict[str, float | int] = {
        "name_exact_match": _exact_match(s1_name, cand_name),
        "address_exact_match": _exact_match(s1_address, cand_address),
        "country_exact_match": 1 if str(s1_country).strip().lower() == str(cand_country).strip().lower() and str(s1_country).strip() else 0,
        "name_fuzz_ratio": fuzz.ratio(s1_name_norm, cand_name_norm),
        "name_partial_ratio": fuzz.partial_ratio(s1_name_norm, cand_name_norm),
        "name_token_sort_ratio": fuzz.token_sort_ratio(s1_name_norm, cand_name_norm),
        "name_token_set_ratio": fuzz.token_set_ratio(s1_name_norm, cand_name_norm),
        "address_fuzz_ratio": fuzz.ratio(s1_addr_norm, cand_addr_norm),
        "address_partial_ratio": fuzz.partial_ratio(s1_addr_norm, cand_addr_norm),
        "address_token_sort_ratio": fuzz.token_sort_ratio(s1_addr_norm, cand_addr_norm),
        "address_token_set_ratio": fuzz.token_set_ratio(s1_addr_norm, cand_addr_norm),
        "name_length_difference": _length_difference(s1_name, cand_name),
        "address_length_difference": _length_difference(s1_address, cand_address),
        "name_token_overlap": _token_overlap(s1_name, cand_name),
        "address_token_overlap": _token_overlap(s1_address, cand_address),
        "exact_normalized_name": 1 if s1_name_norm and s1_name_norm == cand_name_norm else 0,
        "exact_normalized_address": 1 if s1_addr_norm and s1_addr_norm == cand_addr_norm else 0,
    }
    return features


def load_candidate_pairs(path: str) -> List[Tuple[str, str]]:
    """Read the future Member 1 candidate-pair file.

    Expected output schema:
      source1_entity_id <tab> candidate_entity_ids

    candidate_entity_ids is comma-separated, possibly empty.
    """
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        actual = set((reader.fieldnames or []))
        if not required.issubset(actual):
            missing = sorted(required - actual)
            raise ValueError(f"Candidate file missing required columns: {missing}")

        for row in reader:
            source1_id = str(row.get("source1_entity_id", "")).strip()
            if not source1_id:
                continue
            raw_candidates = row.get("candidate_entity_ids", "") or ""
            candidates = [candidate.strip() for candidate in str(raw_candidates).split(",") if candidate.strip()]
            for candidate_id in candidates:
                rows.append((source1_id, candidate_id))
    return rows


def candidate_pairs_to_map(path: str) -> Dict[str, List[str]]:
    """Map each source1 entity to the list of candidate IDs in the file."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    for source1_id, candidate_id in load_candidate_pairs(path):
        mapping[source1_id].append(candidate_id)
    return {key: list(dict.fromkeys(value)) for key, value in mapping.items()}


def parse_matched_ids(raw_value: object) -> set[str]:
    if raw_value is None:
        return set()
    text = str(raw_value).strip()
    if text == "" or text.lower() == "nan":
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def build_pair_labels(candidate_pairs: Iterable[Tuple[str, str]], ground_truth_map: Mapping[str, Iterable[str]]) -> List[Tuple[str, str, int]]:
    """Create (source1_entity_id, candidate_entity_id, label) labels.

    If the candidate appears in the source1 ground-truth list, label is 1,
    otherwise 0. This allows zero, one, or multiple true matches per source1.
    """
    labels: List[Tuple[str, str, int]] = []
    for source1_id, candidate_id in candidate_pairs:
        true_ids = set(ground_truth_map.get(source1_id, []))
        label = 1 if candidate_id in true_ids else 0
        labels.append((source1_id, candidate_id, label))
    return labels


def load_ground_truth_map(path: str) -> Dict[str, set[str]]:
    """Read source1_entity_id -> set(matched_entity_ids)."""
    mapping: Dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        actual = set((reader.fieldnames or []))
        if not required.issubset(actual):
            missing = sorted(required - actual)
            raise ValueError(f"Ground-truth file missing required columns: {missing}")
        for row in reader:
            source1_id = str(row.get("source1_entity_id", "")).strip()
            if not source1_id:
                continue
            mapping[source1_id] = parse_matched_ids(row.get("matched_entity_ids"))
    return mapping
