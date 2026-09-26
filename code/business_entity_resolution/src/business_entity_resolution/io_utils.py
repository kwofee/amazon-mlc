"""Strict TSV readers shared by scoring, folds, and index construction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Set, Tuple


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _require_columns(fieldnames: List[str] | None, required: List[str], path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path}: missing TSV header")
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def iter_source_rows(path: str | Path) -> Iterator[Dict[str, str]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(reader.fieldnames, SOURCE_COLUMNS, source_path)
        for row_number, row in enumerate(reader, start=2):
            entity_id = (row.get("entity_id") or "").strip()
            if not entity_id:
                raise ValueError(f"{source_path}:{row_number}: empty entity_id")
            yield {column: row.get(column, "") or "" for column in SOURCE_COLUMNS}


def parse_id_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def iter_id_set_rows(
    path: str | Path,
    id_column: str,
    values_column: str,
) -> Iterator[Tuple[str, Set[str]]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(reader.fieldnames, [id_column, values_column], source_path)
        for row_number, row in enumerate(reader, start=2):
            entity_id = (row.get(id_column) or "").strip()
            if not entity_id:
                raise ValueError(f"{source_path}:{row_number}: empty {id_column}")
            values = parse_id_list(row.get(values_column, "") or "")
            if len(values) != len(set(values)):
                raise ValueError(f"{source_path}:{row_number}: duplicate ID within {values_column}")
            yield entity_id, set(values)


def read_id_set_mapping(
    path: str | Path,
    id_column: str,
    values_column: str,
) -> Dict[str, set[str]]:
    mapping: Dict[str, set[str]] = {}
    source_path = Path(path)
    for entity_id, values in iter_id_set_rows(source_path, id_column, values_column):
        if entity_id in mapping:
            raise ValueError(f"{source_path}: duplicate {id_column} {entity_id}")
        mapping[entity_id] = values
    return mapping
