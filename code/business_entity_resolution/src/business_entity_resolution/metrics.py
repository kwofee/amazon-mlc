"""Exact grouped entity-resolution metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
import tempfile
from typing import Mapping, Set

from .io_utils import iter_id_set_rows


@dataclass(frozen=True)
class MacroScore:
    beta: float
    macro_fbeta: float
    anchor_count: int
    singleton_count: int
    correctly_empty_singletons: int
    anchors_with_any_true_positive: int

    def to_dict(self) -> dict:
        return asdict(self)


def entity_fbeta(truth: Set[str], prediction: Set[str], beta: float = 0.5) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not truth:
        return 1.0 if not prediction else 0.0
    if not prediction:
        return 0.0
    true_positive_count = len(truth & prediction)
    if true_positive_count == 0:
        return 0.0
    precision = true_positive_count / len(prediction)
    recall = true_positive_count / len(truth)
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


def macro_entity_fbeta(
    truth_by_anchor: Mapping[str, Set[str]],
    prediction_by_anchor: Mapping[str, Set[str]],
    beta: float = 0.5,
    strict_anchor_set: bool = True,
) -> MacroScore:
    truth_ids = set(truth_by_anchor)
    prediction_ids = set(prediction_by_anchor)
    if strict_anchor_set and truth_ids != prediction_ids:
        missing = sorted(truth_ids - prediction_ids)[:5]
        extra = sorted(prediction_ids - truth_ids)[:5]
        raise ValueError(
            "prediction anchor set differs from truth: "
            f"missing={len(truth_ids - prediction_ids)} sample={missing}; "
            f"extra={len(prediction_ids - truth_ids)} sample={extra}"
        )
    if not truth_ids:
        raise ValueError("truth contains no anchors")

    total = 0.0
    singleton_count = 0
    correctly_empty = 0
    anchors_with_tp = 0
    for anchor_id, truth in truth_by_anchor.items():
        prediction = prediction_by_anchor.get(anchor_id, set())
        if not truth:
            singleton_count += 1
            correctly_empty += int(not prediction)
        if truth & prediction:
            anchors_with_tp += 1
        total += entity_fbeta(set(truth), set(prediction), beta=beta)

    return MacroScore(
        beta=beta,
        macro_fbeta=total / len(truth_ids),
        anchor_count=len(truth_ids),
        singleton_count=singleton_count,
        correctly_empty_singletons=correctly_empty,
        anchors_with_any_true_positive=anchors_with_tp,
    )


def score_files(
    truth_path: str | Path,
    predictions_path: str | Path,
    beta: float = 0.5,
) -> MacroScore:
    if beta <= 0:
        raise ValueError("beta must be positive")

    def load_table(
        connection: sqlite3.Connection,
        table: str,
        path: str | Path,
    ) -> int:
        connection.execute(
            f"CREATE TABLE {table} ("
            "anchor_id TEXT PRIMARY KEY, matched_ids TEXT NOT NULL"
            ") WITHOUT ROWID"
        )
        batch = []
        count = 0
        for anchor_id, matched_ids in iter_id_set_rows(
            path,
            id_column="source1_entity_id",
            values_column="matched_entity_ids",
        ):
            batch.append((anchor_id, ",".join(sorted(matched_ids))))
            if len(batch) >= 50_000:
                try:
                    connection.executemany(f"INSERT INTO {table} VALUES (?, ?)", batch)
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"{path}: duplicate source1_entity_id") from error
                count += len(batch)
                batch.clear()
        if batch:
            try:
                connection.executemany(f"INSERT INTO {table} VALUES (?, ?)", batch)
            except sqlite3.IntegrityError as error:
                raise ValueError(f"{path}: duplicate source1_entity_id") from error
            count += len(batch)
        connection.commit()
        return count

    with tempfile.NamedTemporaryFile(prefix="macro-score-", suffix=".sqlite") as temporary:
        connection = sqlite3.connect(temporary.name)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            truth_count = load_table(connection, "truth", truth_path)
            prediction_count = load_table(connection, "prediction", predictions_path)
            missing_count = connection.execute(
                "SELECT COUNT(*) FROM truth t LEFT JOIN prediction p "
                "ON t.anchor_id=p.anchor_id WHERE p.anchor_id IS NULL"
            ).fetchone()[0]
            extra_count = connection.execute(
                "SELECT COUNT(*) FROM prediction p LEFT JOIN truth t "
                "ON p.anchor_id=t.anchor_id WHERE t.anchor_id IS NULL"
            ).fetchone()[0]
            if missing_count or extra_count or truth_count != prediction_count:
                raise ValueError(
                    "prediction anchor set differs from truth: "
                    f"missing={missing_count}; extra={extra_count}; "
                    f"truth={truth_count}; predictions={prediction_count}"
                )
            if truth_count == 0:
                raise ValueError("truth contains no anchors")

            total = 0.0
            singleton_count = 0
            correctly_empty = 0
            anchors_with_tp = 0
            rows = connection.execute(
                "SELECT t.matched_ids AS truth_ids, p.matched_ids AS prediction_ids "
                "FROM truth t JOIN prediction p ON t.anchor_id=p.anchor_id"
            )
            for truth_text, prediction_text in rows:
                truth = set(truth_text.split(",")) if truth_text else set()
                prediction = set(prediction_text.split(",")) if prediction_text else set()
                if not truth:
                    singleton_count += 1
                    correctly_empty += int(not prediction)
                if truth & prediction:
                    anchors_with_tp += 1
                total += entity_fbeta(truth, prediction, beta=beta)

            return MacroScore(
                beta=beta,
                macro_fbeta=total / truth_count,
                anchor_count=truth_count,
                singleton_count=singleton_count,
                correctly_empty_singletons=correctly_empty,
                anchors_with_any_true_positive=anchors_with_tp,
            )
        finally:
            connection.close()
