from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
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

    raw_text = str(value).strip()
    if raw_text.lower() in {"nan", "<na>"}:
        return ""
    text = unicodedata.normalize("NFKC", raw_text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
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


def _token_count_difference(a: object, b: object) -> int:
    return abs(_token_count(a) - _token_count(b))


def _containment(a: str, b: str) -> int:
    return int(bool(a and b and (a in b or b in a)))


def _edge_token_agreement(a: object, b: object, last: bool = False) -> int:
    tokens_a = _safe_tokens(a)
    tokens_b = _safe_tokens(b)
    if not tokens_a or not tokens_b:
        return 0
    return int((tokens_a[-1] == tokens_b[-1]) if last else (tokens_a[0] == tokens_b[0]))


FEATURE_COLUMNS = (
    "name_exact_match",
    "address_exact_match",
    "country_exact_match",
    "country_mismatch",
    "source1_country_missing",
    "candidate_country_missing",
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "address_fuzz_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "name_length_difference",
    "address_length_difference",
    "name_token_count_difference",
    "address_token_count_difference",
    "name_token_overlap",
    "address_token_overlap",
    "name_containment",
    "address_containment",
    "name_first_token_match",
    "name_last_token_match",
    "exact_normalized_name",
    "exact_normalized_address",
)


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    name: str
    address: str
    country: str
    name_tokens: Tuple[str, ...]
    address_tokens: Tuple[str, ...]
    name_token_set: frozenset[str]
    address_token_set: frozenset[str]


def prepare_normalized_record(
    name: str,
    address: str,
    country: str,
    name_tokens: str | None = None,
    address_tokens: str | None = None,
) -> PreparedRecord:
    """Prepare cached token data from normalized fields, optionally stored tokens."""
    prepared_name_tokens = tuple((name_tokens if name_tokens is not None else name).split())
    prepared_address_tokens = tuple(
        (address_tokens if address_tokens is not None else address).split()
    )
    return PreparedRecord(
        name=name,
        address=address,
        country=country,
        name_tokens=prepared_name_tokens,
        address_tokens=prepared_address_tokens,
        name_token_set=frozenset(prepared_name_tokens),
        address_token_set=frozenset(prepared_address_tokens),
    )


def prepare_record(record: Mapping[str, object]) -> PreparedRecord:
    """Normalize and tokenize one entity record for reuse across its pairs."""
    return prepare_normalized_record(
        normalize_text(record.get("business_name", "")),
        normalize_text(record.get("business_address", "")),
        normalize_text(record.get("country", "")),
    )


def _prepared_token_overlap(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    smaller, larger = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    intersection = sum(token in larger for token in smaller)
    union = len(tokens_a) + len(tokens_b) - intersection
    return intersection / union if union else 0.0


def generate_pair_feature_values(
    source1: PreparedRecord, candidate: PreparedRecord
) -> Tuple[float | int, ...]:
    """Return features in FEATURE_COLUMNS order without per-pair dict allocation."""
    name_exact = int(bool(source1.name and source1.name == candidate.name))
    address_exact = int(bool(source1.address and source1.address == candidate.address))
    country_exact = int(bool(source1.country and source1.country == candidate.country))

    return (
        name_exact,
        address_exact,
        country_exact,
        int(bool(source1.country and candidate.country and not country_exact)),
        int(not source1.country),
        int(not candidate.country),
        fuzz.ratio(source1.name, candidate.name),
        fuzz.partial_ratio(source1.name, candidate.name),
        fuzz.token_sort_ratio(source1.name, candidate.name),
        fuzz.token_set_ratio(source1.name, candidate.name),
        fuzz.ratio(source1.address, candidate.address),
        fuzz.partial_ratio(source1.address, candidate.address),
        fuzz.token_sort_ratio(source1.address, candidate.address),
        fuzz.token_set_ratio(source1.address, candidate.address),
        abs(len(source1.name) - len(candidate.name)),
        abs(len(source1.address) - len(candidate.address)),
        abs(len(source1.name_tokens) - len(candidate.name_tokens)),
        abs(len(source1.address_tokens) - len(candidate.address_tokens)),
        _prepared_token_overlap(source1.name_token_set, candidate.name_token_set),
        _prepared_token_overlap(source1.address_token_set, candidate.address_token_set),
        _containment(source1.name, candidate.name),
        _containment(source1.address, candidate.address),
        int(bool(source1.name_tokens and candidate.name_tokens and source1.name_tokens[0] == candidate.name_tokens[0])),
        int(bool(source1.name_tokens and candidate.name_tokens and source1.name_tokens[-1] == candidate.name_tokens[-1])),
        name_exact,
        address_exact,
    )


def generate_pair_features(
    source1_record: Mapping[str, object], candidate_record: Mapping[str, object]
) -> Dict[str, float | int]:
    """Generate the compatible feature mapping for one source1/candidate pair."""
    values = generate_pair_feature_values(
        prepare_record(source1_record), prepare_record(candidate_record)
    )
    return dict(zip(FEATURE_COLUMNS, values))


def iter_candidate_pairs(path: str) -> Iterator[Tuple[str, str]]:
    """Yield candidate pairs from Member 1's grouped TSV without retaining them."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_ids"}
        actual = set((reader.fieldnames or []))
        if not required.issubset(actual):
            raise ValueError(f"Candidate file missing required columns: {sorted(required - actual)}")

        for row in reader:
            source1_id = str(row.get("source1_entity_id", "")).strip()
            if not source1_id:
                continue
            raw_candidates = row.get("candidate_entity_ids", "") or ""
            seen: set[str] = set()
            for raw_candidate in str(raw_candidates).split(","):
                candidate_id = raw_candidate.strip()
                if candidate_id and candidate_id not in seen:
                    seen.add(candidate_id)
                    yield source1_id, candidate_id


def load_candidate_pairs(path: str) -> List[Tuple[str, str]]:
    """Compatibility helper; prefer iter_candidate_pairs for large files."""
    return list(iter_candidate_pairs(path))


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
    truth_sets = {key: set(value) for key, value in ground_truth_map.items()}
    for source1_id, candidate_id in candidate_pairs:
        true_ids = truth_sets.get(source1_id, set())
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
