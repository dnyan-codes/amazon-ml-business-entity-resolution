from __future__ import annotations

import argparse
import csv
import ctypes
import platform
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, Set, Tuple

from features import PreparedRecord, generate_pair_feature_values, prepare_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_CANDIDATE_ROWS = 10_000
DEFAULT_FULL_PAIR_COUNT = 149_959_255
REQUIRED_SOURCE_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}
REQUIRED_CANDIDATE_COLUMNS = {"source1_entity_id", "candidate_entity_ids"}
def iter_candidate_rows(path: Path, limit: int) -> Iterator[Tuple[str, list[str]]]:
    """Read at most `limit` grouped candidate rows, deduplicating IDs per row."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        actual_columns = set(reader.fieldnames or [])
        missing = REQUIRED_CANDIDATE_COLUMNS - actual_columns
        if missing:
            raise ValueError(f"{path} is missing candidate columns: {sorted(missing)}")

        for row_number, row in enumerate(islice(reader, limit), start=1):
            source1_id = str(row.get("source1_entity_id", "")).strip()
            if not source1_id:
                raise ValueError(f"Blank Source 1 ID in candidate row {row_number}.")
            raw_ids = str(row.get("candidate_entity_ids", "") or "")
            candidate_ids = list(
                dict.fromkeys(value.strip() for value in raw_ids.split(",") if value.strip())
            )
            yield source1_id, candidate_ids


def collect_required_ids(
    candidate_path: Path, row_limit: int
) -> Tuple[Set[str], Set[str], Set[str], int, int]:
    source1_ids: Set[str] = set()
    source2_ids: Set[str] = set()
    source3_ids: Set[str] = set()
    row_count = pair_count = 0

    for source1_id, candidate_ids in iter_candidate_rows(candidate_path, row_limit):
        row_count += 1
        source1_ids.add(source1_id)
        pair_count += len(candidate_ids)
        for candidate_id in candidate_ids:
            if candidate_id.startswith("S2-"):
                source2_ids.add(candidate_id)
            elif candidate_id.startswith("S3-"):
                source3_ids.add(candidate_id)
            else:
                raise ValueError(f"Unexpected candidate ID prefix: {candidate_id}")

    if row_count == 0:
        raise ValueError(f"No candidate rows found in {candidate_path}.")
    return source1_ids, source2_ids, source3_ids, row_count, pair_count


def load_requested_records(
    path: Path, requested_ids: Set[str]
) -> Dict[str, PreparedRecord]:
    """Scan a source TSV but retain only rows whose IDs are requested."""
    records: Dict[str, PreparedRecord] = {}
    remaining = set(requested_ids)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        actual_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_SOURCE_COLUMNS - actual_columns
        if missing_columns:
            raise ValueError(f"{path} is missing source columns: {sorted(missing_columns)}")

        if not remaining:
            return records

        for row in reader:
            entity_id = str(row.get("entity_id", "")).strip()
            if entity_id not in remaining:
                continue
            records[entity_id] = prepare_record(row)
            remaining.remove(entity_id)
            if not remaining:
                break

    if remaining:
        examples = ", ".join(sorted(remaining)[:5])
        raise ValueError(f"{len(remaining):,} requested IDs were not found in {path}; examples: {examples}")
    return records


def peak_memory_mb() -> float | None:
    """Return approximate peak resident memory using the platform's stdlib/API."""
    if platform.system() == "Windows":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            process = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
            process.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            succeeded = psapi.GetProcessMemoryInfo(
                process.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if succeeded:
                return counters.PeakWorkingSetSize / (1024 * 1024)
        except (AttributeError, OSError):
            return None
        return None

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return peak / (1024 * 1024)
        return peak / 1024
    except (ImportError, AttributeError, OSError):
        return None


def format_duration(seconds: float) -> str:
    if seconds >= 86_400:
        return f"{seconds / 86_400:,.2f} days ({seconds / 3_600:,.1f} hours)"
    if seconds >= 3_600:
        return f"{seconds / 3_600:,.2f} hours ({seconds / 60:,.1f} minutes)"
    if seconds >= 60:
        return f"{seconds / 60:,.2f} minutes ({seconds:,.1f} seconds)"
    return f"{seconds:,.2f} seconds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Member 2 pair features on at most 10,000 candidate rows."
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=PROJECT_ROOT / "output" / "train_candidate_pairs.tsv",
    )
    parser.add_argument("--rows", type=int, default=MAX_CANDIDATE_ROWS)
    parser.add_argument(
        "--source1",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_source1.tsv",
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
    )
    parser.add_argument("--full-pairs", type=int, default=DEFAULT_FULL_PAIR_COUNT)
    args = parser.parse_args()
    if not 1 <= args.rows <= MAX_CANDIDATE_ROWS:
        parser.error(f"--rows must be between 1 and {MAX_CANDIDATE_ROWS:,}.")
    if args.full_pairs <= 0:
        parser.error("--full-pairs must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate_file.resolve()
    source1_path = args.source1.resolve()
    source2_path = args.source2.resolve()
    source3_path = args.source3.resolve()

    source1_ids, source2_ids, source3_ids, row_count, expected_pairs = collect_required_ids(
        candidate_path, args.rows
    )
    source1_records = load_requested_records(source1_path, source1_ids)
    source2_records = load_requested_records(source2_path, source2_ids)
    source3_records = load_requested_records(source3_path, source3_ids)
    candidate_records = source2_records
    candidate_records.update(source3_records)
    del source2_records, source3_records

    processed_rows = processed_pairs = 0
    started = time.perf_counter()
    for source1_id, candidate_ids in iter_candidate_rows(candidate_path, args.rows):
        source1_record = source1_records[source1_id]
        for candidate_id in candidate_ids:
            generate_pair_feature_values(source1_record, candidate_records[candidate_id])
            processed_pairs += 1
        processed_rows += 1
    feature_seconds = time.perf_counter() - started

    if processed_rows != row_count or processed_pairs != expected_pairs:
        raise RuntimeError("Candidate file changed between the counting and feature passes.")

    pairs_per_second = processed_pairs / feature_seconds if feature_seconds else 0.0
    estimated_seconds = args.full_pairs / pairs_per_second if pairs_per_second else float("inf")
    memory = peak_memory_mb()

    print("Feature benchmark summary")
    print(f"Candidate rows processed: {processed_rows:,} (maximum {MAX_CANDIDATE_ROWS:,})")
    print(f"Candidate pairs processed: {processed_pairs:,}")
    print(f"Feature-generation time: {feature_seconds:,.2f} seconds")
    print(f"Throughput: {pairs_per_second:,.2f} pairs/second")
    if memory is None:
        print("Peak memory: unavailable on this platform")
    else:
        print(f"Approximate peak process memory: {memory:,.1f} MiB")
    print(f"Estimated time for {args.full_pairs:,} pairs: {format_duration(estimated_seconds)}")
    print("Feature matrix written: no")
    print("Model training performed: no")


if __name__ == "__main__":
    main()