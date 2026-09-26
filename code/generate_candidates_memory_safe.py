from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import pandas as pd

# Reuse the team's existing normalization logic.
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
MEMBER1_SRC_DIR = CODE_DIR / "business_entity_resolution" / "src"

import sys

if str(MEMBER1_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER1_SRC_DIR))

from preprocessing import normalize_text


SPLIT_FILES = {
    "train": (
        PROJECT_ROOT / "data" / "train_source1.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
        PROJECT_ROOT / "output" / "train_candidate_pairs.tsv",
    ),
    "val": (
        PROJECT_ROOT / "data" / "val_source1.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
        PROJECT_ROOT / "output" / "val_candidate_pairs.tsv",
    ),
    "test": (
        PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv",
        PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv",
        PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv",
        PROJECT_ROOT / "output" / "candidate_pairs.tsv",
    ),
}

DEFAULT_MAX_CANDIDATES = 100
CHUNK_SIZE = 100_000


def make_key(country: str, value: str, length: int) -> str:
    return f"{country}|{value[:length]}"


def create_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")

    conn.execute(
        """
        CREATE TABLE candidate_index (
            block_key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_candidate_block
        ON candidate_index(block_key)
        """
    )

    return conn


def index_source(
    conn: sqlite3.Connection,
    source_path: Path,
    source_label: str,
) -> None:
    print(f"Indexing {source_label}: {source_path}")

    total = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    ):
        rows = []

        for row in chunk.itertuples(index=False):
            entity_id = str(row.entity_id)

            name = normalize_text(row.business_name)
            address = normalize_text(row.business_address)
            country = normalize_text(row.country)

            name_key = make_key(country, name, 6)
            address_key = make_key(country, address, 8)

            rows.append((name_key, entity_id))
            rows.append((address_key, entity_id))

        conn.executemany(
            """
            INSERT INTO candidate_index
            (block_key, entity_id)
            VALUES (?, ?)
            """,
            rows,
        )

        conn.commit()

        total += len(chunk)

        print(
            f"{source_label}: indexed {total:,} records",
            flush=True,
        )


def get_candidates(
    conn: sqlite3.Connection,
    country: str,
    name: str,
    address: str,
    max_candidates: int,
) -> list[str]:

    keys = [
        make_key(country, name, 6),
        make_key(country, address, 8),
    ]

    candidates = []

    for key in keys:
        rows = conn.execute(
            """
            SELECT entity_id
            FROM candidate_index
            WHERE block_key = ?
            """,
            (key,),
        )

        for (entity_id,) in rows:
            if entity_id not in candidates:
                candidates.append(entity_id)

            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def generate_candidates(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    output_path: Path,
    db_path: Path,
    max_candidates: int,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if db_path.exists():
        db_path.unlink()

    print("Creating disk-backed blocking index...")
    conn = create_database(db_path)

    try:
        index_source(
            conn,
            source2_path,
            "Source 2",
        )

        index_source(
            conn,
            source3_path,
            "Source 3",
        )

        print()
        print("Processing Source 1...")

        total = 0

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file,
                delimiter="\t",
            )

            writer.writerow(
                [
                    "source1_entity_id",
                    "candidate_entity_ids",
                ]
            )

            for chunk in pd.read_csv(
                source1_path,
                sep="\t",
                chunksize=CHUNK_SIZE,
                dtype=str,
                keep_default_na=False,
            ):

                for row in chunk.itertuples(index=False):

                    entity_id = str(row.entity_id)

                    name = normalize_text(
                        row.business_name
                    )

                    address = normalize_text(
                        row.business_address
                    )

                    country = normalize_text(
                        row.country
                    )

                    candidates = get_candidates(
                        conn,
                        country,
                        name,
                        address,
                        max_candidates,
                    )

                    writer.writerow(
                        [
                            entity_id,
                            ",".join(candidates),
                        ]
                    )

                    total += 1

                    if total % 10_000 == 0:
                        print(
                            f"Processed "
                            f"{total:,} Source-1 entities",
                            flush=True,
                        )

        print()
        print("Candidate generation completed.")
        print(f"Rows written: {total:,}")
        print(f"Output: {output_path}")

    finally:
        conn.close()

        if db_path.exists():
            db_path.unlink()

        print("Temporary SQLite index removed.")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Memory-safe candidate generation "
            "using a disk-backed SQLite index."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=tuple(SPLIT_FILES),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum number of candidates per Source-1 entity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output candidate-pairs TSV path.",
    )

    args = parser.parse_args()
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be greater than 0")
    max_candidates = args.max_candidates

    (
        source1_path,
        source2_path,
        source3_path,
        output_path,
    ) = SPLIT_FILES[args.split]
    if args.output is not None:
        output_path = args.output

    for path in (
        source1_path,
        source2_path,
        source3_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

    db_path = (
        PROJECT_ROOT
        / "output"
        / f".candidate_index_{args.split}.db"
    )

    print(f"Generating candidates for: {args.split}")
    print(f"Source 1: {source1_path}")
    print(f"Source 2: {source2_path}")
    print(f"Source 3: {source3_path}")
    print(f"Maximum candidates: {max_candidates}")

    generate_candidates(
        source1_path,
        source2_path,
        source3_path,
        output_path,
        db_path,
        max_candidates,
    )


if __name__ == "__main__":
    main()