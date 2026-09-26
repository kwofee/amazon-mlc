"""Pairwise feature extraction for anchor-target pairs."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .normalization import (
    LEGAL_NAME_TOKENS,
    address_views,
    character_ngrams,
    compact_alphanumeric,
    core_name_tokens,
    name_views,
    text_tokens,
    useful_address_tokens,
)

FEATURE_NAMES: Tuple[str, ...] = (
    "name_exact_full",
    "name_exact_core",
    "name_exact_sorted",
    "name_token_jaccard",
    "name_core_token_jaccard",
    "name_char_trigram_dice",
    "name_char_4gram_dice",
    "name_edit_ratio",
    "name_length_ratio",
    "name_shared_core_count",
    "name_core_token_count_anchor",
    "name_core_token_count_target",
    "name_acronym_match",
    "addr_exact_full",
    "addr_token_jaccard",
    "addr_useful_token_jaccard",
    "addr_char_trigram_dice",
    "addr_length_ratio",
    "addr_shared_useful_count",
    "addr_useful_token_count_anchor",
    "addr_useful_token_count_target",
    "num_shared_count",
    "num_anchor_count",
    "num_target_count",
    "num_jaccard",
    "postal_match",
    "postal_either_present",
    "target_addr_missing",
    "is_s2",
    "script_mismatch",
    "route_count",
)


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def _has_latin(text: str) -> bool:
    return any(
        ("A" <= c <= "ɏ") or ("Ḁ" <= c <= "ỿ")
        for c in text
    )


def _script_bucket(text: str) -> str:
    latin = _has_latin(text)
    deva = _has_devanagari(text)
    if latin and deva:
        return "mixed"
    if deva:
        return "devanagari"
    if latin:
        return "latin"
    has_alpha = any(c.isalpha() for c in text)
    return "other" if has_alpha else "none"


def _jaccard(a: FrozenSet[str] | set, b: FrozenSet[str] | set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dice(a: FrozenSet[str] | set, b: FrozenSet[str] | set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    denom = len(a) + len(b)
    return 2.0 * inter / denom if denom else 0.0


def _edit_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    distance = prev[la]
    max_len = max(la, lb)
    return 1.0 - distance / max_len


def _length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


def extract_pair_features(
    anchor_name: str,
    anchor_address: str,
    anchor_country: str,
    target_name: str,
    target_address: str,
    target_source: str,
    route_count: int = 0,
) -> Dict[str, float]:
    a_names = name_views(anchor_name)
    a_addrs = address_views(anchor_address, anchor_country)
    t_names = name_views(target_name)
    t_addrs = address_views(target_address, anchor_country)

    a_core = set(a_names["name_tokens_core"])
    t_core = set(t_names["name_tokens_core"])
    a_full = set(a_names["name_tokens_full"])
    t_full = set(t_names["name_tokens_full"])

    a_addr_useful = set(a_addrs["address_tokens_useful"])
    t_addr_useful = set(t_addrs["address_tokens_useful"])
    a_addr_full = set(a_addrs["address_tokens_full"])
    t_addr_full = set(t_addrs["address_tokens_full"])

    a_nums = a_addrs["address_numeric_set"]
    t_nums = t_addrs["address_numeric_set"]

    a_trigrams = character_ngrams(anchor_name, 3, 3)
    t_trigrams = character_ngrams(target_name, 3, 3)
    a_4grams = a_names["name_char_ngrams"]
    t_4grams = t_names["name_char_ngrams"]

    a_addr_trigrams = character_ngrams(anchor_address, 3, 3)
    t_addr_trigrams = character_ngrams(target_address, 3, 3)

    a_postal = a_addrs["postal_code"]
    t_postal = t_addrs["postal_code"]
    postal_either = 1.0 if (a_postal or t_postal) else 0.0
    postal_match = 0.0
    if a_postal and t_postal:
        postal_match = 1.0 if a_postal.value == t_postal.value else -1.0

    target_addr_missing = 1.0 if not target_address.strip() or target_address.strip().lower() == "null" else 0.0

    a_script = _script_bucket(anchor_name)
    t_script = _script_bucket(target_name)
    script_mismatch = 0.0 if a_script == t_script else 1.0

    acronym_match = 0.0
    if a_names["name_acronym"] and t_names["name_acronym"]:
        if a_names["name_acronym"] == t_names["name_acronym"]:
            acronym_match = 1.0

    return {
        "name_exact_full": float(a_names["name_norm_full"] == t_names["name_norm_full"] and bool(a_names["name_norm_full"])),
        "name_exact_core": float(a_names["name_core_ordered"] == t_names["name_core_ordered"] and bool(a_names["name_core_ordered"])),
        "name_exact_sorted": float(a_names["name_core_sorted"] == t_names["name_core_sorted"] and bool(a_names["name_core_sorted"])),
        "name_token_jaccard": _jaccard(a_full, t_full),
        "name_core_token_jaccard": _jaccard(a_core, t_core),
        "name_char_trigram_dice": _dice(a_trigrams, t_trigrams),
        "name_char_4gram_dice": _dice(a_4grams, t_4grams),
        "name_edit_ratio": _edit_ratio(a_names["name_core_ordered"], t_names["name_core_ordered"]),
        "name_length_ratio": _length_ratio(a_names["name_core_ordered"], t_names["name_core_ordered"]),
        "name_shared_core_count": float(len(a_core & t_core)),
        "name_core_token_count_anchor": float(len(a_core)),
        "name_core_token_count_target": float(len(t_core)),
        "name_acronym_match": acronym_match,
        "addr_exact_full": float(
            a_addrs["address_norm_full"] == t_addrs["address_norm_full"]
            and bool(a_addrs["address_norm_full"])
            and not target_addr_missing
        ),
        "addr_token_jaccard": _jaccard(a_addr_full, t_addr_full),
        "addr_useful_token_jaccard": _jaccard(a_addr_useful, t_addr_useful),
        "addr_char_trigram_dice": _dice(a_addr_trigrams, t_addr_trigrams),
        "addr_length_ratio": _length_ratio(
            " ".join(a_addrs["address_tokens_useful"]),
            " ".join(t_addrs["address_tokens_useful"]),
        ),
        "addr_shared_useful_count": float(len(a_addr_useful & t_addr_useful)),
        "addr_useful_token_count_anchor": float(len(a_addr_useful)),
        "addr_useful_token_count_target": float(len(t_addr_useful)),
        "num_shared_count": float(len(a_nums & t_nums)),
        "num_anchor_count": float(len(a_nums)),
        "num_target_count": float(len(t_nums)),
        "num_jaccard": _jaccard(a_nums, t_nums),
        "postal_match": postal_match,
        "postal_either_present": postal_either,
        "target_addr_missing": target_addr_missing,
        "is_s2": float(target_source == "S2"),
        "script_mismatch": script_mismatch,
        "route_count": float(route_count),
    }
