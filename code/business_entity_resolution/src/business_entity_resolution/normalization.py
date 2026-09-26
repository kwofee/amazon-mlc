"""Multi-view text normalization and Phase 1 numeric-address signatures."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


LEGAL_NAME_TOKENS: FrozenSet[str] = frozenset(
    {
        "inc", "incorporated", "corp", "corporation", "co", "company",
        "llc", "llp", "ltd", "limited", "plc", "pvt", "private",
        "sarl", "sas", "sa", "sasu", "eurl", "sci", "gmbh",
        "group", "holdings", "enterprise", "enterprises", "the", "and",
        "of", "dba",
    }
)

ADDRESS_LOW_INFORMATION_TOKENS: FrozenSet[str] = frozenset(
    {
        "road", "rd", "street", "st", "avenue", "ave", "lane", "ln",
        "drive", "dr", "boulevard", "blvd", "highway", "hwy", "route",
        "rue", "de", "la", "le", "du", "near", "opposite", "opp",
        "behind", "at", "in", "and", "the", "no", "number", "district",
        "dist", "state", "city", "county", "nagar", "colony", "phase",
        "sector", "null",
    }
)

_NUMBER_BEARING_RE = re.compile(
    r"#?[^\W_]+(?:[-/][^\W_]+)*",
    flags=re.UNICODE,
)
_ORDINAL_RE = re.compile(r"^(\d+)(?:ST|ND|RD|TH)$", flags=re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+)",
    flags=re.IGNORECASE,
)
_NON_ADDRESS_IDENTIFIER_CONTEXT = re.compile(
    r"\b(?:phone|mobile|mob|telephone|tel|fax|gst|gstin|cin|tax|registration|regn|account|acct)\b",
    flags=re.IGNORECASE,
)
_POSTAL_EXCLUDED_CONTEXT = re.compile(
    r"\b(?:phone|mobile|mob|telephone|tel|fax|gst|gstin|cin|tax|registration|regn|account|acct|survey|khasra|sector|phase)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PostalCode:
    value: str
    confidence: float
    cue: str
    explicit: bool


@dataclass(frozen=True)
class NumericAddressViews:
    tokens_ordered: Tuple[str, ...]
    signature_ordered: str
    signature_sorted: str
    token_set: FrozenSet[str]
    postal_code: Optional[PostalCode]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["token_set"] = sorted(self.token_set)
        return result


def _ascii_decimal_digits(text: str) -> str:
    converted: List[str] = []
    for character in text:
        if character.isdecimal():
            try:
                converted.append(str(unicodedata.digit(character)))
                continue
            except (TypeError, ValueError):
                pass
        converted.append(character)
    return "".join(converted)


def normalize_text(text: str, accent_fold: bool = False) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = _ascii_decimal_digits(value).casefold().replace("&", " and ")
    if accent_fold:
        value = "".join(
            character
            for character in unicodedata.normalize("NFKD", value)
            if unicodedata.category(character) != "Mn"
        )
    return " ".join(_unicode_word_tokens(value))


def _unicode_word_tokens(text: str) -> List[str]:
    """Tokenize letters/numbers while retaining combining marks in Indic words."""
    tokens: List[str] = []
    current: List[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isalnum() or (category.startswith("M") and current):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def text_tokens(text: str, accent_fold: bool = False) -> List[str]:
    normalized = normalize_text(text, accent_fold=accent_fold)
    return normalized.split() if normalized else []


def core_name_tokens(text: str) -> List[str]:
    return [token for token in text_tokens(text) if token not in LEGAL_NAME_TOKENS]


def useful_address_tokens(text: str) -> List[str]:
    return [
        token
        for token in text_tokens(text)
        if token not in ADDRESS_LOW_INFORMATION_TOKENS
    ]


def compact_alphanumeric(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or unicodedata.category(character).startswith("M")
    )


def domain_stem(text: str) -> str:
    match = _DOMAIN_RE.search((text or "").casefold())
    if not match:
        return ""
    hostname = match.group(1).strip(".")
    labels = hostname.split(".")
    if len(labels) < 2:
        return ""
    return labels[-2].replace("-", "")


def character_ngrams(text: str, minimum: int = 3, maximum: int = 5) -> FrozenSet[str]:
    compact = compact_alphanumeric(text)
    grams = {
        compact[start : start + size]
        for size in range(minimum, maximum + 1)
        for start in range(0, max(0, len(compact) - size + 1))
    }
    return frozenset(grams)


def name_views(text: str) -> Dict[str, object]:
    full_tokens = text_tokens(text)
    core_tokens = [token for token in full_tokens if token not in LEGAL_NAME_TOKENS]
    return {
        "name_norm_full": " ".join(full_tokens),
        "name_norm_accent_folded": normalize_text(text, accent_fold=True),
        "name_tokens_full": tuple(full_tokens),
        "name_tokens_core": tuple(core_tokens),
        "name_core_ordered": " ".join(core_tokens),
        "name_core_sorted": "|".join(sorted(core_tokens)),
        "name_compact_alnum": compact_alphanumeric(text),
        "name_domain_stem": domain_stem(text),
        "name_acronym": "".join(token[0] for token in core_tokens if token),
        "name_char_ngrams": character_ngrams(text),
    }


def _canonical_numeric_token(raw_token: str) -> str:
    token = _ascii_decimal_digits(unicodedata.normalize("NFKC", raw_token)).upper()
    token = token.lstrip("#")
    token = re.sub(r"\s+", "", token)
    ordinal = _ORDINAL_RE.match(token)
    if ordinal:
        return ordinal.group(1)
    return token.replace("-", "")


def _postal_patterns(country: str) -> Tuple[re.Pattern[str], re.Pattern[str]]:
    country_key = (country or "").casefold()
    if country_key == "india":
        explicit = re.compile(
            r"\b(?P<cue>pin(?:\s*code)?|postal(?:\s*code)?)\s*[:#-]?\s*(?P<code>\d{6})\b",
            flags=re.IGNORECASE,
        )
        bare = re.compile(r"(?<!\d)(?P<code>\d{6})(?!\d)")
    elif country_key == "us":
        explicit = re.compile(
            r"\b(?P<cue>zip(?:\s*code)?|postal(?:\s*code)?)\s*[:#-]?\s*(?P<code>\d{5}(?:-\d{4})?)\b",
            flags=re.IGNORECASE,
        )
        bare = re.compile(r"(?<!\d)(?P<code>\d{5}(?:-\d{4})?)(?!\d)")
    elif country_key == "france":
        explicit = re.compile(
            r"\b(?P<cue>code\s*postal|postal(?:\s*code)?)\s*[:#-]?\s*(?P<code>\d{5})\b",
            flags=re.IGNORECASE,
        )
        bare = re.compile(r"(?<!\d)(?P<code>\d{5})(?!\d)")
    else:
        explicit = re.compile(
            r"\b(?P<cue>postal(?:\s*code)?|zip(?:\s*code)?|pin(?:\s*code)?)\s*[:#-]?\s*(?P<code>\d{4,10}(?:-\d{2,4})?)\b",
            flags=re.IGNORECASE,
        )
        bare = re.compile(r"(?!)")
    return explicit, bare


def extract_postal_code(address: str, country: str) -> Optional[PostalCode]:
    normalized = _ascii_decimal_digits(unicodedata.normalize("NFKC", address or ""))
    explicit_pattern, bare_pattern = _postal_patterns(country)
    explicit_match = explicit_pattern.search(normalized)
    if explicit_match:
        return PostalCode(
            value=explicit_match.group("code").upper(),
            confidence=1.0,
            cue=explicit_match.group("cue").casefold(),
            explicit=True,
        )

    matches = list(bare_pattern.finditer(normalized))
    for match in reversed(matches):
        context = normalized[max(0, match.start() - 24) : match.start()]
        if _POSTAL_EXCLUDED_CONTEXT.search(context):
            continue
        if match.start() >= max(0, int(len(normalized) * 0.55)):
            return PostalCode(
                value=match.group("code").upper(),
                confidence=0.6,
                cue="country-format-near-end",
                explicit=False,
            )
    return None


def numeric_address_views(address: str, country: str) -> NumericAddressViews:
    normalized = _ascii_decimal_digits(unicodedata.normalize("NFKC", address or ""))
    postal_code = extract_postal_code(normalized, country)
    numeric_tokens: List[str] = []
    for match in _NUMBER_BEARING_RE.finditer(normalized):
        raw_token = match.group(0)
        if not any(character.isdigit() for character in raw_token):
            continue
        context = normalized[max(0, match.start() - 24) : match.start()]
        if _NON_ADDRESS_IDENTIFIER_CONTEXT.search(context):
            continue
        canonical = _canonical_numeric_token(raw_token)
        if not canonical:
            continue
        if postal_code and postal_code.explicit:
            postal_compact = postal_code.value.replace("-", "")
            if canonical.replace("-", "") == postal_compact:
                continue
        numeric_tokens.append(canonical)

    ordered = tuple(numeric_tokens)
    return NumericAddressViews(
        tokens_ordered=ordered,
        signature_ordered="|".join(ordered),
        signature_sorted="|".join(sorted(ordered)),
        token_set=frozenset(ordered),
        postal_code=postal_code,
    )


def address_views(text: str, country: str) -> Dict[str, object]:
    full_tokens = text_tokens(text)
    useful_tokens = [
        token for token in full_tokens if token not in ADDRESS_LOW_INFORMATION_TOKENS
    ]
    numeric = numeric_address_views(text, country)
    return {
        "address_norm_full": " ".join(
            token for token in full_tokens if token != "null"
        ),
        "address_tokens_full": tuple(full_tokens),
        "address_tokens_useful": tuple(useful_tokens),
        "address_sorted_multiset": "|".join(sorted(useful_tokens)),
        "address_char_ngrams": character_ngrams(text),
        "address_numeric_tokens_ordered": numeric.tokens_ordered,
        "address_numeric_signature_ordered": numeric.signature_ordered,
        "address_numeric_signature_sorted": numeric.signature_sorted,
        "address_numeric_set": numeric.token_set,
        "postal_code": numeric.postal_code,
    }
