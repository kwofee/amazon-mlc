"""Deterministic, grouped S1-anchor validation-fold generation."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .io_utils import iter_source_rows, parse_id_list


FOLD_COLUMNS = [
    "source1_entity_id",
    "fold",
    "country",
    "match_count",
    "s2_count",
    "s3_count",
    "source_composition",
    "singleton",
]


def stable_fold(anchor_id: str, stratum: str, fold_count: int, seed: str) -> int:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    payload = f"{seed}\x1f{stratum}\x1f{anchor_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % fold_count


def match_profile(match_ids: Sequence[str]) -> Tuple[int, int, int, str, int]:
    s2_count = sum(entity_id.startswith("S2-") for entity_id in match_ids)
    s3_count = sum(entity_id.startswith("S3-") for entity_id in match_ids)
    invalid = [
        entity_id
        for entity_id in match_ids
        if not entity_id.startswith("S2-") and not entity_id.startswith("S3-")
    ]
    if invalid:
        raise ValueError(f"ground truth contains non-S2/S3 IDs: {invalid[:5]}")
    if not match_ids:
        composition = "none"
    elif s2_count and s3_count:
        composition = "both"
    elif s2_count:
        composition = "s2_only"
    else:
        composition = "s3_only"
    return len(match_ids), s2_count, s3_count, composition, int(not match_ids)


def _initialize_country_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE anchor_country ("
        "entity_id TEXT PRIMARY KEY, country TEXT NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE seen_ground_truth ("
        "entity_id TEXT PRIMARY KEY"
        ") WITHOUT ROWID"
    )


def _load_anchor_countries(connection: sqlite3.Connection, source1_path: str | Path) -> int:
    batch: List[Tuple[str, str]] = []
    count = 0
    for row in iter_source_rows(source1_path):
        batch.append((row["entity_id"], row["country"]))
        if len(batch) >= 50_000:
            connection.executemany("INSERT INTO anchor_country VALUES (?, ?)", batch)
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO anchor_country VALUES (?, ?)", batch)
        count += len(batch)
    connection.commit()
    return count


def _country_map(
    connection: sqlite3.Connection,
    anchor_ids: Sequence[str],
) -> Dict[str, str]:
    placeholders = ",".join("?" for _ in anchor_ids)
    query = f"SELECT entity_id, country FROM anchor_country WHERE entity_id IN ({placeholders})"
    return dict(connection.execute(query, list(anchor_ids)))


def generate_grouped_folds(
    source1_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    fold_count: int = 5,
    seed: str = "amazon-ml-2026-v1",
) -> Dict[str, int]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="anchor-country-", suffix=".sqlite") as temporary:
        connection = sqlite3.connect(temporary.name)
        try:
            _initialize_country_database(connection)
            source1_count = _load_anchor_countries(connection, source1_path)
            fold_counts = {str(fold): 0 for fold in range(fold_count)}
            ground_truth_count = 0

            with Path(ground_truth_path).open(
                "r", encoding="utf-8", newline=""
            ) as truth_handle, destination.open(
                "w", encoding="utf-8", newline=""
            ) as output_handle:
                reader = csv.DictReader(truth_handle, delimiter="\t")
                required = {"source1_entity_id", "matched_entity_ids"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(
                        f"{ground_truth_path}: expected columns {sorted(required)}"
                    )
                writer = csv.DictWriter(
                    output_handle,
                    fieldnames=FOLD_COLUMNS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()

                pending: List[Dict[str, str]] = []

                def flush_pending() -> None:
                    nonlocal ground_truth_count
                    if not pending:
                        return
                    anchor_ids = [row["source1_entity_id"] for row in pending]
                    try:
                        connection.executemany(
                            "INSERT INTO seen_ground_truth VALUES (?)",
                            [(anchor_id,) for anchor_id in anchor_ids],
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            "ground truth contains a duplicate source1_entity_id"
                        ) from error
                    countries = _country_map(connection, anchor_ids)
                    missing = [anchor_id for anchor_id in anchor_ids if anchor_id not in countries]
                    if missing:
                        raise ValueError(
                            f"ground-truth anchors missing from S1: {missing[:5]}"
                        )
                    for row in pending:
                        anchor_id = row["source1_entity_id"]
                        match_ids = parse_id_list(row["matched_entity_ids"])
                        match_count, s2_count, s3_count, composition, singleton = (
                            match_profile(match_ids)
                        )
                        country = countries[anchor_id]
                        stratum = f"{country}|{match_count}|{composition}"
                        fold = stable_fold(anchor_id, stratum, fold_count, seed)
                        writer.writerow(
                            {
                                "source1_entity_id": anchor_id,
                                "fold": fold,
                                "country": country,
                                "match_count": match_count,
                                "s2_count": s2_count,
                                "s3_count": s3_count,
                                "source_composition": composition,
                                "singleton": singleton,
                            }
                        )
                        fold_counts[str(fold)] += 1
                        ground_truth_count += 1
                    pending.clear()

                for row_number, row in enumerate(reader, start=2):
                    anchor_id = (row.get("source1_entity_id") or "").strip()
                    if not anchor_id:
                        raise ValueError(
                            f"{ground_truth_path}:{row_number}: empty source1_entity_id"
                        )
                    pending.append(
                        {
                            "source1_entity_id": anchor_id,
                            "matched_entity_ids": row.get("matched_entity_ids", "") or "",
                        }
                    )
                    if len(pending) >= 800:
                        flush_pending()
                flush_pending()

            if ground_truth_count != source1_count:
                raise ValueError(
                    "S1 and ground-truth row counts differ: "
                    f"source1={source1_count}, ground_truth={ground_truth_count}"
                )
            seen_count = connection.execute(
                "SELECT COUNT(*) FROM seen_ground_truth"
            ).fetchone()[0]
            if seen_count != source1_count:
                raise ValueError(
                    "not every S1 anchor appeared exactly once in ground truth: "
                    f"source1={source1_count}, unique_ground_truth={seen_count}"
                )
            return {
                "source1_count": source1_count,
                "ground_truth_count": ground_truth_count,
                **{f"fold_{key}": value for key, value in fold_counts.items()},
            }
        finally:
            connection.close()
