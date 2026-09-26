"""Candidate pair generation, model training, and prediction."""

from __future__ import annotations

import csv
import json
import logging
import pickle
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .features import FEATURE_NAMES, extract_pair_features
from .indexing import TargetIndex
from .io_utils import iter_source_rows, parse_id_list
from .normalization import address_views, name_views

LOGGER = logging.getLogger(__name__)


def _numeric_suffix(entity_id: str) -> int:
    return int(entity_id.split("-", 1)[1])


def _blocker_candidates(
    index: TargetIndex,
    country: str,
    source: str,
    names: Dict[str, object],
    addresses: Dict[str, object],
    max_df: int,
    rare_limit: int,
    exact_cap: int,
    bm25_name_limit: int,
    bm25_address_limit: int,
    bm25_max_token_df: int,
    numeric_limit: int,
    ngram_name_limit: int,
) -> Dict[str, Set[str]]:
    """Run all retrieval routes and return {route: set_of_target_ids}."""
    from .recall import (
        _bm25_route_candidates,
        _exact_route_candidates,
        _ngram_route_candidates,
        _numeric_route_candidates,
        _rare_route_candidates,
    )

    routes: Dict[str, Set[str]] = {}
    exact = _exact_route_candidates(index, country, source, names, addresses, exact_cap)
    routes.update(exact)
    rare = _rare_route_candidates(
        index, country, source, names, addresses, max_df, rare_limit
    )
    routes.update(rare)
    bm25 = _bm25_route_candidates(
        index, country, source, names, addresses,
        bm25_name_limit, bm25_address_limit, max_token_df=bm25_max_token_df,
    )
    routes.update(bm25)
    numeric = _numeric_route_candidates(
        index, country, source, addresses, numeric_limit,
    )
    routes.update(numeric)
    ngram = _ngram_route_candidates(
        index, country, source, names, ngram_name_limit,
    )
    routes.update(ngram)
    return routes


def generate_training_pairs(
    index_paths: Sequence[str | Path],
    fold_path: str | Path,
    source1_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    fold: int = 0,
    sample_modulo: Optional[int] = None,
    sample_remainder: int = 0,
    max_df: int = 5_000,
    rare_limit: int = 50,
    exact_cap: int = 500,
    bm25_name_limit: int = 20,
    bm25_address_limit: int = 10,
    bm25_max_token_df: int = 5_000,
    numeric_limit: int = 50,
    ngram_name_limit: int = 20,
    negatives_per_anchor: int = 20,
    progress_every: int = 500,
) -> dict:
    """Generate labeled (anchor, target, features, label) pairs from the blocker."""

    # Load fold anchors
    selected: Dict[str, Dict[str, object]] = {}
    with Path(fold_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["fold"]) != fold:
                continue
            aid = row["source1_entity_id"]
            if sample_modulo and _numeric_suffix(aid) % sample_modulo != sample_remainder:
                continue
            selected[aid] = {"country": row["country"]}

    # Load S1 records
    remaining = set(selected)
    for row in iter_source_rows(source1_path):
        if row["entity_id"] in remaining:
            selected[row["entity_id"]]["business_name"] = row["business_name"]
            selected[row["entity_id"]]["business_address"] = row["business_address"]
            remaining.discard(row["entity_id"])

    # Load ground truth
    remaining = set(selected)
    with Path(ground_truth_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            aid = row["source1_entity_id"]
            if aid in remaining:
                selected[aid]["truth"] = set(parse_id_list(row.get("matched_entity_ids", "")))
                remaining.discard(aid)

    # Open indexes and generate pairs
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_positives = 0
    total_negatives = 0
    missed_positives = 0

    with ExitStack() as stack:
        source_indexes: Dict[str, TargetIndex] = {}
        for ip in index_paths:
            opened = stack.enter_context(TargetIndex(ip))
            for src in opened.sources():
                source_indexes[src] = opened

        # Check that raw_records are available in at least one index
        has_raw = any(idx.has_raw_records() for idx in source_indexes.values())
        if not has_raw:
            raise RuntimeError(
                "No raw_records table found in indexes. "
                "Run 'populate-raw-records' first."
            )

        writer_f = out.open("w", encoding="utf-8", newline="")
        stack.callback(writer_f.close)
        fieldnames = ["anchor_id", "target_id", "label"] + list(FEATURE_NAMES)
        writer = csv.DictWriter(writer_f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for pos, (anchor_id, rec) in enumerate(selected.items(), 1):
            country = str(rec["country"])
            anchor_name = str(rec.get("business_name", ""))
            anchor_addr = str(rec.get("business_address", ""))
            truth: Set[str] = rec.get("truth", set())
            names = name_views(anchor_name)
            addresses = address_views(anchor_addr, country)

            # Run blocker for both sources
            all_candidates: Dict[str, int] = {}  # target_id -> route_count
            for source in ("S2", "S3"):
                if source not in source_indexes:
                    continue
                routes = _blocker_candidates(
                    source_indexes[source], country, source, names, addresses,
                    max_df, rare_limit, exact_cap,
                    bm25_name_limit, bm25_address_limit, bm25_max_token_df,
                    numeric_limit, ngram_name_limit,
                )
                target_routes: Dict[str, int] = defaultdict(int)
                for route, tids in routes.items():
                    for tid in tids:
                        target_routes[tid] += 1
                for tid, rc in target_routes.items():
                    all_candidates[tid] = all_candidates.get(tid, 0) + rc

            # Separate positives and negatives
            retrieved_positives = truth & set(all_candidates)
            negatives = set(all_candidates) - truth

            # Sample negatives
            neg_list = sorted(negatives)
            if len(neg_list) > negatives_per_anchor:
                neg_list.sort(key=lambda t: -all_candidates[t])
                neg_list = neg_list[:negatives_per_anchor]

            # Batch-fetch target records from SQLite
            needed_ids = sorted(retrieved_positives) + neg_list
            target_records: Dict[str, Dict[str, str]] = {}
            for source, idx in source_indexes.items():
                source_ids = [t for t in needed_ids if t.startswith(source)]
                if source_ids:
                    target_records.update(idx.get_records_batch(source_ids))

            # Write positive pairs
            for tid in sorted(retrieved_positives):
                trec = target_records.get(tid)
                if not trec:
                    continue
                t_source = "S2" if tid.startswith("S2") else "S3"
                feats = extract_pair_features(
                    anchor_name, anchor_addr, country,
                    trec["business_name"], trec["business_address"],
                    t_source, route_count=all_candidates.get(tid, 0),
                )
                row_out = {"anchor_id": anchor_id, "target_id": tid, "label": 1}
                row_out.update(feats)
                writer.writerow(row_out)
                total_positives += 1

            # Count missed positives (truth not retrieved by blocker)
            missed_positives += len(truth - retrieved_positives)

            # Write negative pairs
            for tid in neg_list:
                trec = target_records.get(tid)
                if not trec:
                    continue
                t_source = "S2" if tid.startswith("S2") else "S3"
                feats = extract_pair_features(
                    anchor_name, anchor_addr, country,
                    trec["business_name"], trec["business_address"],
                    t_source, route_count=all_candidates.get(tid, 0),
                )
                row_out = {"anchor_id": anchor_id, "target_id": tid, "label": 0}
                row_out.update(feats)
                writer.writerow(row_out)
                total_negatives += 1

            if progress_every and pos % progress_every == 0:
                LOGGER.info(
                    "processed %d anchors, %d positives, %d negatives so far",
                    pos, total_positives, total_negatives,
                )

    result = {
        "anchors": len(selected),
        "positives": total_positives,
        "negatives": total_negatives,
        "missed_positives": missed_positives,
        "output": str(out),
    }
    LOGGER.info("pair generation complete: %s", json.dumps(result))
    return result


def train_matcher(
    train_pairs_path: str | Path,
    model_output_path: str | Path,
    val_pairs_path: Optional[str | Path] = None,
    max_iter: int = 500,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    min_samples_leaf: int = 50,
) -> dict:
    """Train a HistGradientBoostingClassifier on generated pairs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    LOGGER.info("loading training pairs from %s", train_pairs_path)
    X_train, y_train, ids_train = _load_pairs(train_pairs_path)
    LOGGER.info("training set: %d pairs (%d positive, %d negative)",
                len(y_train), sum(y_train), len(y_train) - sum(y_train))

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        early_stopping=val_pairs_path is not None,
        validation_fraction=0.1 if val_pairs_path is None else None,
        verbose=1,
    )
    model.fit(X_train, y_train)

    result: Dict[str, object] = {
        "train_samples": len(y_train),
        "train_positives": int(sum(y_train)),
        "features": list(FEATURE_NAMES),
        "n_iter": int(model.n_iter_),
    }

    # Train metrics
    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = model.predict(X_train)
    result["train_metrics"] = {
        "auc": round(float(roc_auc_score(y_train, train_proba)), 4),
        "avg_precision": round(float(average_precision_score(y_train, train_proba)), 4),
        "precision": round(float(precision_score(y_train, train_pred)), 4),
        "recall": round(float(recall_score(y_train, train_pred)), 4),
        "f1": round(float(f1_score(y_train, train_pred)), 4),
    }

    # Validation metrics
    if val_pairs_path:
        LOGGER.info("loading validation pairs from %s", val_pairs_path)
        X_val, y_val, ids_val = _load_pairs(val_pairs_path)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = model.predict(X_val)
        result["val_samples"] = len(y_val)
        result["val_positives"] = int(sum(y_val))
        result["val_metrics"] = {
            "auc": round(float(roc_auc_score(y_val, val_proba)), 4),
            "avg_precision": round(float(average_precision_score(y_val, val_proba)), 4),
            "precision": round(float(precision_score(y_val, val_pred)), 4),
            "recall": round(float(recall_score(y_val, val_pred)), 4),
            "f1": round(float(f1_score(y_val, val_pred)), 4),
        }

    # Save model
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    result["model_path"] = str(model_path)

    LOGGER.info("training complete: %s", json.dumps(result))
    return result


def _load_pairs(path: str | Path) -> Tuple[List[List[float]], List[int], List[Tuple[str, str]]]:
    X: List[List[float]] = []
    y: List[int] = []
    ids: List[Tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            features = [float(row[name]) for name in FEATURE_NAMES]
            X.append(features)
            y.append(int(row["label"]))
            ids.append((row["anchor_id"], row["target_id"]))
    return X, y, ids
