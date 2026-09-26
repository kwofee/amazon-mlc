"""Disk-backed exact, rare-token, BM25, character n-gram, and numeric target indexes."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .io_utils import iter_source_rows
from .normalization import address_views, character_ngrams, compact_alphanumeric, name_views


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExactLookupResult:
    bucket_size: int
    entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RareCandidate:
    entity_id: str
    rarest_document_frequency: int
    supporting_token_count: int
    supporting_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class BM25Candidate:
    entity_id: str
    score: float
    matching_token_count: int


@dataclass(frozen=True)
class NgramCandidate:
    entity_id: str
    score: float
    shared_ngram_count: int


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA page_size=32768")
    journal_mode = connection.execute("PRAGMA journal_mode=MEMORY").fetchone()[0]
    LOGGER.info("SQLite journal mode: %s", journal_mode)
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA mmap_size=4294967296")


def initialize_index_database(connection: sqlite3.Connection) -> None:
    _configure_connection(connection)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS exact_postings (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            view TEXT NOT NULL,
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (country, source, view, key, entity_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS token_postings (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            token TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (country, source, field, token, entity_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS token_stats (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            token TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            PRIMARY KEY (country, source, field, token)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS ngram_postings (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            ngram TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (country, source, field, ngram, entity_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS ngram_stats (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            ngram TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            PRIMARY KEY (country, source, field, ngram)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS raw_records (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,
            country TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    connection.commit()


def source_from_entity_id(entity_id: str) -> str:
    if entity_id.startswith("S2-"):
        return "S2"
    if entity_id.startswith("S3-"):
        return "S3"
    raise ValueError(f"target entity_id does not start with S2-/S3-: {entity_id}")


def _useful_numeric_signature(tokens: Sequence[str]) -> bool:
    if len(tokens) >= 2:
        return True
    if len(tokens) == 1:
        token = tokens[0]
        return any(character.isalpha() for character in token) and any(
            character.isdigit() for character in token
        )
    return False


def _record_postings(row: Dict[str, str]) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    entity_id = row["entity_id"]
    country = row["country"]
    source = source_from_entity_id(entity_id)
    names = name_views(row["business_name"])
    addresses = address_views(row["business_address"], country)

    exact_values = {
        "name_full": names["name_norm_full"],
        "name_core": names["name_core_ordered"],
        "name_sorted": names["name_core_sorted"],
        "name_domain": names["name_domain_stem"],
        "address_full": addresses["address_norm_full"],
    }
    numeric_tokens = addresses["address_numeric_tokens_ordered"]
    if _useful_numeric_signature(numeric_tokens):
        exact_values["numeric_ordered"] = addresses[
            "address_numeric_signature_ordered"
        ]
        exact_values["numeric_sorted"] = addresses[
            "address_numeric_signature_sorted"
        ]
    postal = addresses["postal_code"]
    if postal is not None and postal.confidence >= 0.9:
        exact_values["postal"] = postal.value

    exact_rows = [
        (country, source, view, str(key), entity_id)
        for view, key in exact_values.items()
        if key
    ]

    token_fields = {
        "name_core": set(names["name_tokens_core"]),
        "address_useful": set(addresses["address_tokens_useful"]),
        "numeric": set(addresses["address_numeric_tokens_ordered"]),
    }
    token_rows = [
        (country, source, field, str(token), entity_id)
        for field, tokens in token_fields.items()
        for token in tokens
        if token
    ]
    return exact_rows, token_rows


def build_target_index(
    target_paths: Sequence[str | Path],
    database_path: str | Path,
    batch_posting_count: int = 100_000,
    overwrite: bool = False,
    resume: bool = False,
    maximum_rows_per_file: Optional[int] = None,
    skip_stats: bool = False,
) -> Dict[str, int]:
    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if len(target_paths) != 1:
            raise ValueError("resume currently requires exactly one target file")
        if not destination.exists() or destination.stat().st_size == 0:
            raise FileNotFoundError(f"cannot resume missing index database: {destination}")
    elif destination.exists() and destination.stat().st_size > 0:
        if not overwrite:
            raise FileExistsError(
                f"index database already exists: {destination}; pass overwrite=True to replace it"
            )
        destination.unlink()
        for sidecar_suffix in ("-wal", "-shm", "-journal"):
            sidecar = Path(f"{destination}{sidecar_suffix}")
            if sidecar.exists():
                sidecar.unlink()
    connection = sqlite3.connect(destination)
    initialize_index_database(connection)
    exact_batch: List[Tuple[str, ...]] = []
    token_batch: List[Tuple[str, ...]] = []
    row_count = 0
    already_indexed = 0
    if resume:
        already_indexed = connection.execute(
            "SELECT COUNT(*) FROM exact_postings WHERE view='name_full'"
        ).fetchone()[0]
        row_count = already_indexed
        LOGGER.info("resuming after %s committed target rows", f"{already_indexed:,}")
    try:
        for target_path in target_paths:
            LOGGER.info("indexing target file %s", target_path)
            for file_row_number, row in enumerate(iter_source_rows(target_path), start=1):
                if file_row_number <= already_indexed:
                    continue
                if (
                    maximum_rows_per_file is not None
                    and file_row_number > maximum_rows_per_file
                ):
                    break
                exact_rows, token_rows = _record_postings(row)
                exact_batch.extend(exact_rows)
                token_batch.extend(token_rows)
                row_count += 1
                if row_count % 100_000 == 0:
                    LOGGER.info("normalized and indexed %s target rows", f"{row_count:,}")
                if len(exact_batch) + len(token_batch) >= batch_posting_count:
                    exact_batch.sort()
                    token_batch.sort()
                    connection.executemany(
                        "INSERT INTO exact_postings VALUES (?, ?, ?, ?, ?)",
                        exact_batch,
                    )
                    connection.executemany(
                        "INSERT INTO token_postings VALUES (?, ?, ?, ?, ?)",
                        token_batch,
                    )
                    connection.commit()
                    exact_batch.clear()
                    token_batch.clear()

        if exact_batch:
            exact_batch.sort()
            connection.executemany(
                "INSERT INTO exact_postings VALUES (?, ?, ?, ?, ?)",
                exact_batch,
            )
        if token_batch:
            token_batch.sort()
            connection.executemany(
                "INSERT INTO token_postings VALUES (?, ?, ?, ?, ?)",
                token_batch,
            )
        connection.commit()

        if skip_stats:
            LOGGER.info("skipping token_stats materialization (skip_stats=True)")
        else:
            LOGGER.info("materializing token document frequencies")
            connection.execute("DELETE FROM token_stats")
            connection.execute(
                "INSERT INTO token_stats "
                "SELECT country, source, field, token, COUNT(*) "
                "FROM token_postings GROUP BY country, source, field, token"
            )
        metadata = {
            "schema_version": "1",
            "target_row_count": str(row_count),
            "target_paths": json.dumps([str(Path(path)) for path in target_paths]),
            "maximum_rows_per_file": json.dumps(maximum_rows_per_file),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO build_metadata VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        LOGGER.info("token document frequencies materialized")

        exact_count = connection.execute(
            "SELECT COUNT(*) FROM exact_postings"
        ).fetchone()[0]
        token_count = connection.execute(
            "SELECT COUNT(*) FROM token_postings"
        ).fetchone()[0]
        token_stat_count = connection.execute(
            "SELECT COUNT(*) FROM token_stats"
        ).fetchone()[0]
        return {
            "target_rows": row_count,
            "exact_postings": exact_count,
            "token_postings": token_count,
            "token_stats": token_stat_count,
        }
    finally:
        connection.close()


def build_ngram_index(
    target_paths: Sequence[str | Path],
    database_path: str | Path,
    max_ngram_df: int = 1000,
    ngram_min: int = 4,
    ngram_max: int = 5,
    maximum_rows_per_file: Optional[int] = None,
) -> Dict[str, int]:
    """Build character n-gram postings for name retrieval, stored alongside the main index.

    Two-pass approach: first count n-gram document frequencies, then only store
    n-grams with df <= max_ngram_df to keep the index compact.
    """
    destination = Path(database_path)
    if not destination.exists():
        raise FileNotFoundError(f"base index must exist: {destination}")

    connection = sqlite3.connect(destination)
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA cache_size=-524288")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS ngram_postings (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            ngram TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (country, source, field, ngram, entity_id)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS ngram_stats (
            country TEXT NOT NULL,
            source TEXT NOT NULL,
            field TEXT NOT NULL,
            ngram TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            PRIMARY KEY (country, source, field, ngram)
        ) WITHOUT ROWID;
        """
    )
    connection.execute("DELETE FROM ngram_postings")
    connection.execute("DELETE FROM ngram_stats")
    connection.commit()

    LOGGER.info("pass 1: counting n-gram document frequencies")
    ngram_counts: DefaultDict[Tuple[str, str, str, str], int] = defaultdict(int)
    row_count = 0
    for target_path in target_paths:
        for file_row_number, row in enumerate(iter_source_rows(target_path), start=1):
            if maximum_rows_per_file is not None and file_row_number > maximum_rows_per_file:
                break
            entity_id = row["entity_id"]
            country = row["country"]
            source = source_from_entity_id(entity_id)
            name = row["business_name"]
            grams = character_ngrams(name, minimum=ngram_min, maximum=ngram_max)
            for gram in grams:
                ngram_counts[(country, source, "name", gram)] += 1
            row_count += 1
            if row_count % 500_000 == 0:
                LOGGER.info("pass 1: counted n-grams for %s rows", f"{row_count:,}")

    LOGGER.info("pass 1 done: %s rows, %s distinct (country,source,field,ngram) keys",
                f"{row_count:,}", f"{len(ngram_counts):,}")

    eligible = {
        key for key, count in ngram_counts.items()
        if count <= max_ngram_df
    }
    LOGGER.info("eligible n-grams (df <= %d): %s of %s",
                max_ngram_df, f"{len(eligible):,}", f"{len(ngram_counts):,}")

    stat_batch = [
        (country, source, field, ngram, ngram_counts[(country, source, field, ngram)])
        for (country, source, field, ngram) in eligible
    ]
    stat_batch.sort()
    connection.executemany(
        "INSERT INTO ngram_stats VALUES (?, ?, ?, ?, ?)", stat_batch
    )
    connection.commit()
    del stat_batch

    LOGGER.info("pass 2: writing eligible n-gram postings")
    posting_batch: List[Tuple[str, ...]] = []
    row_count2 = 0
    postings_written = 0
    for target_path in target_paths:
        for file_row_number, row in enumerate(iter_source_rows(target_path), start=1):
            if maximum_rows_per_file is not None and file_row_number > maximum_rows_per_file:
                break
            entity_id = row["entity_id"]
            country = row["country"]
            source = source_from_entity_id(entity_id)
            name = row["business_name"]
            grams = character_ngrams(name, minimum=ngram_min, maximum=ngram_max)
            for gram in grams:
                key = (country, source, "name", gram)
                if key in eligible:
                    posting_batch.append((country, source, "name", gram, entity_id))
            row_count2 += 1
            if len(posting_batch) >= 200_000:
                posting_batch.sort()
                connection.executemany(
                    "INSERT OR IGNORE INTO ngram_postings VALUES (?, ?, ?, ?, ?)",
                    posting_batch,
                )
                postings_written += len(posting_batch)
                connection.commit()
                posting_batch.clear()
            if row_count2 % 500_000 == 0:
                LOGGER.info("pass 2: processed %s rows, %s postings written",
                            f"{row_count2:,}", f"{postings_written:,}")

    if posting_batch:
        posting_batch.sort()
        connection.executemany(
            "INSERT OR IGNORE INTO ngram_postings VALUES (?, ?, ?, ?, ?)",
            posting_batch,
        )
        postings_written += len(posting_batch)
        connection.commit()

    LOGGER.info("n-gram index complete: %s postings, %s eligible n-grams",
                f"{postings_written:,}", f"{len(eligible):,}")
    connection.close()
    return {
        "target_rows": row_count,
        "eligible_ngrams": len(eligible),
        "ngram_postings": postings_written,
    }


def populate_raw_records(
    database_path: str | Path,
    target_paths: Sequence[str | Path],
    batch_size: int = 50_000,
) -> dict:
    """Insert raw target records into the raw_records table."""
    db = Path(database_path)
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS raw_records ("
        "entity_id TEXT PRIMARY KEY, "
        "business_name TEXT NOT NULL, "
        "business_address TEXT NOT NULL, "
        "country TEXT NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.commit()

    total = 0
    batch: List[Tuple[str, str, str, str]] = []
    for tp in target_paths:
        LOGGER.info("loading raw records from %s", tp)
        for row in iter_source_rows(tp):
            batch.append((
                row["entity_id"],
                row["business_name"],
                row["business_address"],
                row["country"],
            ))
            if len(batch) >= batch_size:
                connection.executemany(
                    "INSERT OR IGNORE INTO raw_records VALUES (?,?,?,?)", batch
                )
                connection.commit()
                total += len(batch)
                batch.clear()
                if total % 500_000 == 0:
                    LOGGER.info("  inserted %s raw records", f"{total:,}")
    if batch:
        connection.executemany(
            "INSERT OR IGNORE INTO raw_records VALUES (?,?,?,?)", batch
        )
        connection.commit()
        total += len(batch)

    count = connection.execute("SELECT COUNT(*) FROM raw_records").fetchone()[0]
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.close()
    LOGGER.info("raw_records populated: %s rows", f"{count:,}")
    return {"raw_records": count, "inserted": total}


class TargetIndex:
    def __init__(self, database_path: str | Path):
        self.connection = sqlite3.connect(Path(database_path))
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TargetIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def get_record(self, entity_id: str) -> Optional[Dict[str, str]]:
        """Look up a single raw target record by entity_id."""
        row = self.connection.execute(
            "SELECT entity_id, business_name, business_address, country "
            "FROM raw_records WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_records_batch(self, entity_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        """Look up multiple raw target records. Returns {entity_id: record_dict}."""
        if not entity_ids:
            return {}
        result: Dict[str, Dict[str, str]] = {}
        chunk_size = 500
        for start in range(0, len(entity_ids), chunk_size):
            chunk = entity_ids[start:start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.connection.execute(
                f"SELECT entity_id, business_name, business_address, country "
                f"FROM raw_records WHERE entity_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                result[row["entity_id"]] = dict(row)
        return result

    def has_raw_records(self) -> bool:
        """Check if raw_records table exists and has data."""
        try:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM raw_records LIMIT 1"
            ).fetchone()
            return row[0] > 0
        except sqlite3.OperationalError:
            return False

    def sources(self) -> Tuple[str, ...]:
        rows = self.connection.execute(
            "SELECT DISTINCT source FROM exact_postings ORDER BY source"
        )
        return tuple(row["source"] for row in rows)

    def exact_lookup(
        self,
        country: str,
        source: str,
        view: str,
        key: str,
        limit: int = 100,
    ) -> ExactLookupResult:
        parameters = (country, source, view, key)
        bucket_size = self.connection.execute(
            "SELECT COUNT(*) FROM exact_postings "
            "WHERE country=? AND source=? AND view=? AND key=?",
            parameters,
        ).fetchone()[0]
        rows = self.connection.execute(
            "SELECT entity_id FROM exact_postings "
            "WHERE country=? AND source=? AND view=? AND key=? "
            "ORDER BY entity_id LIMIT ?",
            (*parameters, limit),
        )
        return ExactLookupResult(
            bucket_size=bucket_size,
            entity_ids=tuple(row["entity_id"] for row in rows),
        )

    def corpus_size(self, country: str, source: str) -> int:
        """Number of distinct entity_ids indexed for this country/source."""
        row = self.connection.execute(
            "SELECT COUNT(*) FROM exact_postings "
            "WHERE country=? AND source=? AND view='name_full'",
            (country, source),
        ).fetchone()
        return row[0] if row else 0

    def rare_token_candidates(
        self,
        country: str,
        source: str,
        field: str,
        tokens: Iterable[str],
        maximum_document_frequency: int,
        limit: int = 100,
    ) -> Tuple[RareCandidate, ...]:
        token_values = sorted(set(token for token in tokens if token))
        if not token_values:
            return tuple()
        token_frequencies: List[Tuple[str, int]] = []
        for token in token_values:
            row = self.connection.execute(
                "SELECT document_frequency FROM token_stats "
                "WHERE country=? AND source=? AND field=? AND token=?",
                (country, source, field, token),
            ).fetchone()
            if row is not None and row["document_frequency"] <= maximum_document_frequency:
                token_frequencies.append((token, row["document_frequency"]))
        token_frequencies.sort(key=lambda item: (item[1], item[0]))

        support: DefaultDict[str, List[Tuple[str, int]]] = defaultdict(list)
        for token, document_frequency in token_frequencies:
            rows = self.connection.execute(
                "SELECT entity_id FROM token_postings "
                "WHERE country=? AND source=? AND field=? AND token=?",
                (country, source, field, token),
            )
            for row in rows:
                support[row["entity_id"]].append((token, document_frequency))

        candidates = [
            RareCandidate(
                entity_id=entity_id,
                rarest_document_frequency=min(frequency for _, frequency in evidence),
                supporting_token_count=len(evidence),
                supporting_tokens=tuple(sorted(token for token, _ in evidence)),
            )
            for entity_id, evidence in support.items()
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.rarest_document_frequency,
                -candidate.supporting_token_count,
                candidate.entity_id,
            )
        )
        return tuple(candidates[:limit])

    def bm25_candidates(
        self,
        country: str,
        source: str,
        field: str,
        query_tokens: Iterable[str],
        limit: int = 20,
        k1: float = 1.2,
        b: float = 0.75,
        minimum_score: float = 0.0,
        max_token_df: int = 500_000,
    ) -> Tuple[BM25Candidate, ...]:
        """BM25 ranking over the existing token postings and stats.

        Skips tokens with df > max_token_df to avoid scanning huge posting lists
        for common tokens that contribute little discriminative signal.
        """
        tokens = sorted(set(t for t in query_tokens if t))
        if not tokens:
            return ()
        n = self.corpus_size(country, source)
        if n == 0:
            return ()

        placeholders = ",".join("?" for _ in tokens)
        stats_rows = self.connection.execute(
            f"SELECT token, document_frequency FROM token_stats "
            f"WHERE country=? AND source=? AND field=? "
            f"AND token IN ({placeholders})",
            (country, source, field, *tokens),
        ).fetchall()

        token_dfs = [
            (row["token"], row["document_frequency"])
            for row in stats_rows
            if 0 < row["document_frequency"] <= max_token_df
        ]
        if not token_dfs:
            return ()

        token_idfs = {}
        for token, df in token_dfs:
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            if idf > 0:
                token_idfs[token] = idf

        if not token_idfs:
            return ()

        entity_scores: DefaultDict[str, float] = defaultdict(float)
        entity_hits: Counter[str] = Counter()
        for token, idf in token_idfs.items():
            rows = self.connection.execute(
                "SELECT entity_id FROM token_postings "
                "WHERE country=? AND source=? AND field=? AND token=?",
                (country, source, field, token),
            )
            for row in rows:
                eid = row["entity_id"]
                entity_scores[eid] += idf
                entity_hits[eid] += 1

        candidates = [
            BM25Candidate(
                entity_id=eid,
                score=score,
                matching_token_count=entity_hits[eid],
            )
            for eid, score in entity_scores.items()
            if score >= minimum_score
        ]
        candidates.sort(key=lambda c: (-c.score, c.entity_id))
        return tuple(candidates[:limit])

    def numeric_overlap_candidates(
        self,
        country: str,
        source: str,
        anchor_numeric_tokens: Sequence[str],
        anchor_useful_address_tokens: Sequence[str],
        limit: int = 50,
        min_shared_numbers: int = 2,
        locality_required: bool = True,
    ) -> Tuple[BM25Candidate, ...]:
        """Retrieve targets sharing multiple address numbers plus a locality token."""
        numeric_set = set(t for t in anchor_numeric_tokens if t)
        if len(numeric_set) < min_shared_numbers:
            return ()

        entity_num_hits: DefaultDict[str, Set[str]] = defaultdict(set)
        for token in numeric_set:
            rows = self.connection.execute(
                "SELECT entity_id FROM token_postings "
                "WHERE country=? AND source=? AND field='numeric' AND token=?",
                (country, source, token),
            )
            for row in rows:
                entity_num_hits[row["entity_id"]].add(token)

        qualifying = {
            eid: shared
            for eid, shared in entity_num_hits.items()
            if len(shared) >= min_shared_numbers
        }
        if not qualifying:
            return ()

        if not locality_required:
            candidates = [
                BM25Candidate(
                    entity_id=eid,
                    score=float(len(shared)),
                    matching_token_count=len(shared),
                )
                for eid, shared in qualifying.items()
            ]
            candidates.sort(key=lambda c: (-c.score, c.entity_id))
            return tuple(candidates[:limit])

        useful_set = set(anchor_useful_address_tokens) - set(anchor_numeric_tokens)
        useful_set -= {"null"}
        if not useful_set:
            candidates = [
                BM25Candidate(
                    entity_id=eid,
                    score=float(len(shared)),
                    matching_token_count=len(shared),
                )
                for eid, shared in qualifying.items()
            ]
            candidates.sort(key=lambda c: (-c.score, c.entity_id))
            return tuple(candidates[:limit])

        entity_addr_hits: Set[str] = set()
        for token in useful_set:
            row = self.connection.execute(
                "SELECT document_frequency FROM token_stats "
                "WHERE country=? AND source=? AND field='address_useful' AND token=?",
                (country, source, token),
            ).fetchone()
            if row is None or row["document_frequency"] > 50_000:
                continue
            rows = self.connection.execute(
                "SELECT entity_id FROM token_postings "
                "WHERE country=? AND source=? AND field='address_useful' AND token=?",
                (country, source, token),
            )
            for r in rows:
                entity_addr_hits.add(r["entity_id"])

        candidates = [
            BM25Candidate(
                entity_id=eid,
                score=float(len(shared)),
                matching_token_count=len(shared),
            )
            for eid, shared in qualifying.items()
            if eid in entity_addr_hits
        ]
        candidates.sort(key=lambda c: (-c.score, c.entity_id))
        return tuple(candidates[:limit])

    def ngram_candidates(
        self,
        country: str,
        source: str,
        field: str,
        query_text: str,
        limit: int = 20,
        ngram_min: int = 4,
        ngram_max: int = 5,
        min_dice: float = 0.3,
    ) -> Tuple[NgramCandidate, ...]:
        """Retrieve candidates by character n-gram Dice similarity."""
        query_grams = character_ngrams(query_text, minimum=ngram_min, maximum=ngram_max)
        if not query_grams:
            return ()

        has_table = self.connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='ngram_stats'"
        ).fetchone()[0]
        if not has_table:
            return ()

        eligible_grams = []
        for gram in query_grams:
            row = self.connection.execute(
                "SELECT document_frequency FROM ngram_stats "
                "WHERE country=? AND source=? AND field=? AND ngram=?",
                (country, source, field, gram),
            ).fetchone()
            if row is not None:
                eligible_grams.append(gram)

        if not eligible_grams:
            return ()

        entity_shared: Counter[str] = Counter()
        for gram in eligible_grams:
            rows = self.connection.execute(
                "SELECT entity_id FROM ngram_postings "
                "WHERE country=? AND source=? AND field=? AND ngram=?",
                (country, source, field, gram),
            )
            for row in rows:
                entity_shared[row["entity_id"]] += 1

        query_size = len(query_grams)
        candidates = []
        for eid, shared_count in entity_shared.items():
            dice = (2.0 * shared_count) / (query_size + shared_count + query_size * 0.1)
            if dice >= min_dice:
                candidates.append(NgramCandidate(
                    entity_id=eid,
                    score=dice,
                    shared_ngram_count=shared_count,
                ))

        candidates.sort(key=lambda c: (-c.score, c.entity_id))
        return tuple(candidates[:limit])
