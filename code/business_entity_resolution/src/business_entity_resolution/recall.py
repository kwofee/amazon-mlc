"""Candidate-recall evaluation for the exact and rare-token blocker."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

from .indexing import TargetIndex
from .io_utils import iter_source_rows, parse_id_list
from .normalization import address_views, character_ngrams, name_views


def _numeric_suffix(entity_id: str) -> int:
    try:
        return int(entity_id.split("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"entity ID has no numeric suffix: {entity_id}") from error


def _select_anchor_ids(
    fold_path: str | Path,
    fold: int,
    sample_modulo: Optional[int],
    sample_remainder: int,
) -> Dict[str, Dict[str, str]]:
    selected: Dict[str, Dict[str, str]] = {}
    with Path(fold_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "fold", "country"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{fold_path}: expected columns {sorted(required)}")
        for row in reader:
            if int(row["fold"]) != fold:
                continue
            anchor_id = row["source1_entity_id"]
            if sample_modulo is not None:
                if sample_modulo <= 0:
                    raise ValueError("sample_modulo must be positive")
                if _numeric_suffix(anchor_id) % sample_modulo != sample_remainder:
                    continue
            selected[anchor_id] = {"country": row["country"]}
    if not selected:
        raise ValueError("fold/sample selection produced no anchors")
    return selected


def _load_selected_source1(
    selected: MutableMapping[str, Dict[str, object]],
    source1_path: str | Path,
) -> None:
    remaining = set(selected)
    for row in iter_source_rows(source1_path):
        anchor_id = row["entity_id"]
        if anchor_id not in remaining:
            continue
        selected[anchor_id]["business_name"] = row["business_name"]
        selected[anchor_id]["business_address"] = row["business_address"]
        if selected[anchor_id]["country"] != row["country"]:
            raise ValueError(f"country mismatch for {anchor_id}")
        remaining.remove(anchor_id)
    if remaining:
        raise ValueError(f"selected anchors missing from S1: {sorted(remaining)[:5]}")


def _load_selected_truth(
    selected: MutableMapping[str, Dict[str, object]],
    ground_truth_path: str | Path,
) -> None:
    remaining = set(selected)
    with Path(ground_truth_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "matched_entity_ids"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{ground_truth_path}: expected columns {sorted(required)}")
        for row in reader:
            anchor_id = row["source1_entity_id"]
            if anchor_id not in remaining:
                continue
            selected[anchor_id]["truth"] = set(parse_id_list(row["matched_entity_ids"]))
            remaining.remove(anchor_id)
    if remaining:
        raise ValueError(
            f"selected anchors missing from ground truth: {sorted(remaining)[:5]}"
        )


def _quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _exact_route_candidates(
    index: TargetIndex,
    country: str,
    source: str,
    names: Mapping[str, object],
    addresses: Mapping[str, object],
    exact_bucket_cap: int,
) -> Dict[str, Set[str]]:
    view_keys = {
        "exact_name_full": ("name_full", names["name_norm_full"]),
        "exact_name_core": ("name_core", names["name_core_ordered"]),
        "exact_name_sorted": ("name_sorted", names["name_core_sorted"]),
        "exact_domain": ("name_domain", names["name_domain_stem"]),
        "exact_address": ("address_full", addresses["address_norm_full"]),
        "exact_numeric_ordered": (
            "numeric_ordered",
            addresses["address_numeric_signature_ordered"],
        ),
        "exact_numeric_sorted": (
            "numeric_sorted",
            addresses["address_numeric_signature_sorted"],
        ),
    }
    routes: Dict[str, Set[str]] = {}
    for route, (view, key) in view_keys.items():
        if not key:
            continue
        result = index.exact_lookup(
            country, source, view, str(key), limit=exact_bucket_cap + 1
        )
        if result.bucket_size <= exact_bucket_cap:
            routes[route] = set(result.entity_ids)
    return routes


def _rare_route_candidates(
    index: TargetIndex,
    country: str,
    source: str,
    names: Mapping[str, object],
    addresses: Mapping[str, object],
    maximum_document_frequency: int,
    per_route_limit: int,
) -> Dict[str, Set[str]]:
    fields = {
        "rare_name": ("name_core", names["name_tokens_core"]),
        "rare_address": ("address_useful", addresses["address_tokens_useful"]),
        "rare_numeric": ("numeric", addresses["address_numeric_tokens_ordered"]),
    }
    routes: Dict[str, Set[str]] = {}
    for route, (field, tokens) in fields.items():
        candidates = index.rare_token_candidates(
            country=country,
            source=source,
            field=field,
            tokens=tokens,
            maximum_document_frequency=maximum_document_frequency,
            limit=per_route_limit,
        )
        routes[route] = {candidate.entity_id for candidate in candidates}
    return routes


def _bm25_route_candidates(
    index: TargetIndex,
    country: str,
    source: str,
    names: Mapping[str, object],
    addresses: Mapping[str, object],
    bm25_name_limit: int,
    bm25_address_limit: int,
    max_token_df: int = 5_000,
) -> Dict[str, Set[str]]:
    routes: Dict[str, Set[str]] = {}
    name_candidates = index.bm25_candidates(
        country=country,
        source=source,
        field="name_core",
        query_tokens=names["name_tokens_core"],
        limit=bm25_name_limit,
        max_token_df=max_token_df,
    )
    if name_candidates:
        routes["bm25_name"] = {c.entity_id for c in name_candidates}
    addr_candidates = index.bm25_candidates(
        country=country,
        source=source,
        field="address_useful",
        query_tokens=addresses["address_tokens_useful"],
        limit=bm25_address_limit,
        max_token_df=max_token_df,
    )
    if addr_candidates:
        routes["bm25_address"] = {c.entity_id for c in addr_candidates}
    return routes


def _numeric_route_candidates(
    index: TargetIndex,
    country: str,
    source: str,
    addresses: Mapping[str, object],
    numeric_limit: int,
) -> Dict[str, Set[str]]:
    routes: Dict[str, Set[str]] = {}
    candidates = index.numeric_overlap_candidates(
        country=country,
        source=source,
        anchor_numeric_tokens=addresses["address_numeric_tokens_ordered"],
        anchor_useful_address_tokens=addresses["address_tokens_useful"],
        limit=numeric_limit,
        min_shared_numbers=2,
        locality_required=True,
    )
    if candidates:
        routes["numeric_overlap"] = {c.entity_id for c in candidates}

    postal = addresses.get("postal_code")
    if postal is not None and postal.confidence >= 0.9:
        result = index.exact_lookup(country, source, "postal", postal.value, limit=200)
        if result.bucket_size <= 200:
            routes["postal"] = set(result.entity_ids)
    return routes


def _ngram_route_candidates(
    index: TargetIndex,
    country: str,
    source: str,
    names: Mapping[str, object],
    ngram_name_limit: int,
) -> Dict[str, Set[str]]:
    routes: Dict[str, Set[str]] = {}
    name_text = str(names.get("name_core_ordered", ""))
    if not name_text:
        return routes
    candidates = index.ngram_candidates(
        country=country,
        source=source,
        field="name",
        query_text=name_text,
        limit=ngram_name_limit,
        min_dice=0.25,
    )
    if candidates:
        routes["ngram_name"] = {c.entity_id for c in candidates}
    return routes


def evaluate_exact_rare_recall(
    index_paths: Sequence[str | Path],
    fold_path: str | Path,
    source1_path: str | Path,
    ground_truth_path: str | Path,
    fold: int = 0,
    sample_modulo: Optional[int] = None,
    sample_remainder: int = 0,
    maximum_document_frequency: int = 5_000,
    per_rare_route_limit: int = 50,
    exact_bucket_cap: int = 500,
    progress_every: int = 1_000,
    enable_bm25: bool = False,
    bm25_name_limit: int = 20,
    bm25_address_limit: int = 10,
    bm25_max_token_df: int = 5_000,
    enable_numeric: bool = False,
    numeric_limit: int = 50,
    enable_ngram: bool = False,
    ngram_name_limit: int = 20,
) -> dict:
    selected = _select_anchor_ids(
        fold_path, fold, sample_modulo, sample_remainder
    )
    _load_selected_source1(selected, source1_path)
    _load_selected_truth(selected, ground_truth_path)

    total_truth_pairs = 0
    retrieved_truth_pairs = 0
    non_singletons = 0
    any_hit_anchors = 0
    complete_truth_anchors = 0
    singleton_anchors = 0
    candidate_counts: List[int] = []
    route_positive_hits: Counter[str] = Counter()
    route_unique_positive_hits: Counter[str] = Counter()
    route_candidate_counts: Counter[str] = Counter()
    slice_totals: Counter[str] = Counter()
    slice_hits: Counter[str] = Counter()

    with ExitStack() as stack:
        source_indexes: Dict[str, TargetIndex] = {}
        for index_path in index_paths:
            opened = stack.enter_context(TargetIndex(index_path))
            for source in opened.sources():
                if source in source_indexes:
                    raise ValueError(
                        f"source {source} occurs in more than one supplied index"
                    )
                source_indexes[source] = opened
        missing_sources = {"S2", "S3"} - set(source_indexes)
        if missing_sources:
            raise ValueError(f"indexes do not contain sources: {sorted(missing_sources)}")

        for position, (anchor_id, record) in enumerate(selected.items(), start=1):
            country = str(record["country"])
            names = name_views(str(record["business_name"]))
            addresses = address_views(str(record["business_address"]), country)
            truth = set(record["truth"])
            routes: Dict[str, Set[str]] = defaultdict(set)
            for source in ("S2", "S3"):
                index = source_indexes[source]
                source_exact = _exact_route_candidates(
                    index, country, source, names, addresses, exact_bucket_cap
                )
                source_rare = _rare_route_candidates(
                    index,
                    country,
                    source,
                    names,
                    addresses,
                    maximum_document_frequency,
                    per_rare_route_limit,
                )
                all_routes = {**source_exact, **source_rare}
                if enable_bm25:
                    source_bm25 = _bm25_route_candidates(
                        index, country, source, names, addresses,
                        bm25_name_limit, bm25_address_limit,
                        max_token_df=bm25_max_token_df,
                    )
                    all_routes.update(source_bm25)
                if enable_numeric:
                    source_numeric = _numeric_route_candidates(
                        index, country, source, addresses, numeric_limit,
                    )
                    all_routes.update(source_numeric)
                if enable_ngram:
                    source_ngram = _ngram_route_candidates(
                        index, country, source, names, ngram_name_limit,
                    )
                    all_routes.update(source_ngram)
                for route, candidates in all_routes.items():
                    routes[route].update(candidates)

            union: Set[str] = set()
            positive_route_support: Dict[str, Set[str]] = defaultdict(set)
            for route, candidates in routes.items():
                union.update(candidates)
                route_candidate_counts[route] += len(candidates)
                hits = candidates & truth
                route_positive_hits[route] += len(hits)
                for target_id in hits:
                    positive_route_support[target_id].add(route)

            hits = union & truth
            total_truth_pairs += len(truth)
            retrieved_truth_pairs += len(hits)
            candidate_counts.append(len(union))
            if truth:
                non_singletons += 1
                any_hit_anchors += int(bool(hits))
                complete_truth_anchors += int(hits == truth)
            else:
                singleton_anchors += 1

            for target_id, supporting_routes in positive_route_support.items():
                if len(supporting_routes) == 1:
                    route_unique_positive_hits[next(iter(supporting_routes))] += 1

            for target_id in truth:
                source = target_id.split("-", 1)[0]
                for slice_name in (f"country:{country}", f"source:{source}"):
                    slice_totals[slice_name] += 1
                    slice_hits[slice_name] += int(target_id in hits)

            if progress_every and position % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "anchors_processed": position,
                            "pair_recall_so_far": (
                                retrieved_truth_pairs / total_truth_pairs
                                if total_truth_pairs
                                else 0.0
                            ),
                            "mean_candidates_so_far": sum(candidate_counts)
                            / len(candidate_counts),
                        }
                    ),
                    flush=True,
                )

    pair_recall = (
        retrieved_truth_pairs / total_truth_pairs if total_truth_pairs else 0.0
    )
    return {
        "selection": {
            "fold": fold,
            "sample_modulo": sample_modulo,
            "sample_remainder": sample_remainder,
            "anchor_count": len(selected),
        },
        "configuration": {
            "maximum_document_frequency": maximum_document_frequency,
            "per_rare_route_limit": per_rare_route_limit,
            "exact_bucket_cap": exact_bucket_cap,
        },
        "recall": {
            "truth_pairs": total_truth_pairs,
            "retrieved_truth_pairs": retrieved_truth_pairs,
            "pair_recall": pair_recall,
            "non_singleton_anchors": non_singletons,
            "any_hit_anchor_recall": any_hit_anchors / non_singletons
            if non_singletons
            else 0.0,
            "complete_truth_anchor_recall": complete_truth_anchors / non_singletons
            if non_singletons
            else 0.0,
            "singleton_anchors": singleton_anchors,
        },
        "candidate_counts": {
            "mean": sum(candidate_counts) / len(candidate_counts),
            "median": _quantile(candidate_counts, 0.5),
            "p95": _quantile(candidate_counts, 0.95),
            "p99": _quantile(candidate_counts, 0.99),
            "maximum": max(candidate_counts),
        },
        "route_positive_hits": dict(sorted(route_positive_hits.items())),
        "route_unique_positive_hits": dict(sorted(route_unique_positive_hits.items())),
        "mean_candidates_by_route": {
            route: count / len(selected)
            for route, count in sorted(route_candidate_counts.items())
        },
        "slice_pair_recall": {
            name: {
                "truth_pairs": slice_totals[name],
                "retrieved": slice_hits[name],
                "recall": slice_hits[name] / slice_totals[name],
            }
            for name in sorted(slice_totals)
        },
    }
