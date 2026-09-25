# %% [markdown]
# # Candidate generation step 1
#
# This is intentionally a small, readable baseline.
#
# We take a limited number of Source 1 entities from the training ground truth,
# scan all of Sources 2 and 3, and collect candidates through several parallel
# channels. A record only needs to be found by one channel to become a candidate.
#
# This first version does not use fuzzy matching, transliteration, embeddings,
# address-number anchors, or a machine-learning model. Those can be added after
# we measure exactly what this simple baseline misses.

# %%
from collections import defaultdict
from pathlib import Path
import re
import unicodedata

import pandas as pd
from IPython.display import display


# Start small so the complete process is easy to run and inspect.
# Increase this only after you understand the output and runtime.
NUMBER_OF_GROUND_TRUTH_ROWS = 1_000
CHUNK_SIZE = 250_000

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"


# %% [markdown]
# ## Text representations
#
# We do not replace the raw text. Instead, each function creates another view
# of the same value. Different views recover different types of matches.

# %%
def normalize_text(value):
    """Create a conservative Unicode-aware representation of text."""

    # NFKC makes visually equivalent forms more consistent while preserving
    # scripts such as Devanagari, Tamil, Telugu, Gujarati, and Kannada.
    text = unicodedata.normalize("NFKC", str(value)).casefold()

    # Punctuation and repeated spaces should not prevent an otherwise exact
    # match. Python's \w keeps Unicode letters and numbers.
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")

    return " ".join(text.split())


def remove_accents(value):
    """Create a Latin accent-folded view without changing the raw value."""

    # This allows "président" and "president" to meet in one channel.
    # The original accented representation is still available elsewhere.
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return normalize_text(text)


# These words describe company type more often than business identity.
# We remove them only in the core-name view; the complete name is preserved in
# the exact-name channel.
LEGAL_AND_DOMAIN_WORDS = {
    "co", "com", "company", "corp", "corporation", "inc", "llc", "llp",
    "ltd", "limited", "net", "org", "private", "pvt", "sas", "sasu",
    "sarl", "sci", "www",
}


def core_name(value):
    """Create a compact name for spacing, suffix, and domain variations."""

    tokens = normalize_text(value).split()
    identity_tokens = [token for token in tokens if token not in LEGAL_AND_DOMAIN_WORDS]

    # Removing spaces helps pairs such as "Anchor Staking" and
    # "anchorstaking.com" meet in this channel.
    return "".join(identity_tokens)


def country_key(country, value):
    """Keep records from different countries in separate lookup buckets."""

    # A separator creates one ordinary string that Pandas can compare quickly.
    # The country values are read from the data, so unseen labels such as France
    # work without changing this function.
    return f"{country}\x1f{value}"


# %% [markdown]
# ## Load the development Source 1 entities
#
# Ground truth is used here only to choose a small training subset and evaluate
# recall. Candidate generation itself never looks at a target's true label.

# %%
ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=NUMBER_OF_GROUND_TRUTH_ROWS,
)

source1_ids = set(ground_truth["source1_entity_id"])


def read_selected_source1_records():
    """Find the chosen S1 records without loading the full file at once."""

    selected_chunks = []

    for chunk in pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        selected = chunk[chunk["entity_id"].isin(source1_ids)]

        if not selected.empty:
            selected_chunks.append(selected)

    return pd.concat(selected_chunks, ignore_index=True)


source1 = read_selected_source1_records()

print(f"Ground-truth rows selected: {len(ground_truth):,}")
print(f"Source 1 records retrieved: {len(source1):,}")


# %% [markdown]
# ## Build lookup tables for the parallel channels
#
# Each lookup maps a key to every S1 entity that owns that key. We store sets
# because two different S1 businesses can sometimes share the same name.

# %%
exact_name_index = defaultdict(set)
core_name_index = defaultdict(set)
accent_core_name_index = defaultdict(set)
exact_address_index = defaultdict(set)

for record in source1.itertuples(index=False):
    country = record.country
    entity_id = record.entity_id

    exact_name = normalize_text(record.business_name)
    compact_name = core_name(record.business_name)
    accent_compact_name = core_name(remove_accents(record.business_name))
    exact_address = normalize_text(record.business_address)

    if exact_name:
        exact_name_index[country_key(country, exact_name)].add(entity_id)

    if compact_name:
        core_name_index[country_key(country, compact_name)].add(entity_id)

    if accent_compact_name:
        accent_core_name_index[country_key(country, accent_compact_name)].add(entity_id)

    if exact_address:
        exact_address_index[country_key(country, exact_address)].add(entity_id)


# %% [markdown]
# ## Scan Sources 2 and 3
#
# Every channel runs independently. We take the union rather than requiring a
# target record to pass several filters. This is important when a true record
# has a missing address, a domain-style name, or a different component order.

# %%
# One candidate can be found through several channels. Keeping the channel set
# gives us useful debugging information and later becomes a model feature.
candidate_channels = defaultdict(set)


def add_candidates(source1_matches, candidate_id, channel_name):
    """Record all S1 candidates found through one retrieval channel."""

    for source1_id in source1_matches:
        candidate_channels[(source1_id, candidate_id)].add(channel_name)


def collect_channel_matches(keys, candidate_ids, lookup, channel_name):
    """Add only target rows whose prepared key exists in an S1 lookup."""

    # Testing a full Pandas column against the lookup is much faster than
    # running Python lookup code for every one of the millions of target rows.
    matched_mask = keys.isin(lookup)

    for key, candidate_id in zip(keys[matched_mask], candidate_ids[matched_mask]):
        add_candidates(lookup[key], candidate_id, channel_name)


def scan_target_source(path):
    """Scan one complete target source and apply every channel in parallel."""

    print(f"Scanning {path.name} ...")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        exact_names = chunk["business_name"].map(normalize_text)
        compact_names = chunk["business_name"].map(core_name)
        accent_compact_names = chunk["business_name"].map(
            lambda value: core_name(remove_accents(value))
        )
        exact_addresses = chunk["business_address"].map(normalize_text)

        exact_name_keys = chunk["country"] + "\x1f" + exact_names
        core_name_keys = chunk["country"] + "\x1f" + compact_names
        accent_name_keys = chunk["country"] + "\x1f" + accent_compact_names
        exact_address_keys = chunk["country"] + "\x1f" + exact_addresses

        collect_channel_matches(
            exact_name_keys,
            chunk["entity_id"],
            exact_name_index,
            "exact_name",
        )
        collect_channel_matches(
            core_name_keys,
            chunk["entity_id"],
            core_name_index,
            "core_name",
        )
        collect_channel_matches(
            accent_name_keys,
            chunk["entity_id"],
            accent_core_name_index,
            "accent_core_name",
        )
        collect_channel_matches(
            exact_address_keys,
            chunk["entity_id"],
            exact_address_index,
            "exact_address",
        )


scan_target_source(SOURCE2_PATH)
scan_target_source(SOURCE3_PATH)


# %% [markdown]
# ## Inspect and evaluate the candidate list
#
# Blocking recall answers one question: did candidate generation include the
# true links? It does not yet tell us whether a classifier can reject the false
# candidates.

# %%
candidate_details = pd.DataFrame(
    [
        {
            "source1_entity_id": source1_id,
            "candidate_entity_id": candidate_id,
            "channels": ",".join(sorted(channels)),
            "channel_count": len(channels),
        }
        for (source1_id, candidate_id), channels in candidate_channels.items()
    ]
)

true_pairs = set()

for row in ground_truth.itertuples(index=False):
    for matched_id in row.matched_entity_ids.split(","):
        matched_id = matched_id.strip()

        if matched_id:
            true_pairs.add((row.source1_entity_id, matched_id))

retrieved_pairs = set(
    zip(
        candidate_details["source1_entity_id"],
        candidate_details["candidate_entity_id"],
    )
)

retrieved_true_pairs = true_pairs & retrieved_pairs
pair_recall = len(retrieved_true_pairs) / len(true_pairs) if true_pairs else 0.0

candidate_count_by_s1 = (
    candidate_details.groupby("source1_entity_id")
    .size()
    .reindex(ground_truth["source1_entity_id"], fill_value=0)
)

print()
print(f"Candidate pairs created: {len(candidate_details):,}")
print(f"True pairs in selected ground truth: {len(true_pairs):,}")
print(f"True pairs retrieved: {len(retrieved_true_pairs):,}")
print(f"Pair-level blocking recall: {pair_recall:.2%}")
print(f"Average candidates per S1: {candidate_count_by_s1.mean():.1f}")
print(f"S1 entities with no candidates: {(candidate_count_by_s1 == 0).sum():,}")


# %% [markdown]
# ## Save development outputs
#
# These filenames contain "development" because this small training subset is
# for understanding the pipeline. It is not the final test submission.

# %%
OUTPUT_DIR.mkdir(exist_ok=True)

candidate_details.to_csv(
    OUTPUT_DIR / "development_candidate_details.tsv",
    sep="\t",
    index=False,
)

candidate_lists = (
    candidate_details.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
    .agg(lambda values: ",".join(dict.fromkeys(values)))
    .rename("candidate_entity_ids")
    .reset_index()
)

# Every selected S1 must appear, including entities for which the baseline found
# no candidates. This mirrors the required final candidate file structure.
candidate_lists = (
    ground_truth[["source1_entity_id"]]
    .merge(candidate_lists, on="source1_entity_id", how="left")
    .fillna({"candidate_entity_ids": ""})
)

candidate_lists.to_csv(
    OUTPUT_DIR / "development_candidate_pairs.tsv",
    sep="\t",
    index=False,
)

display(candidate_details.head(20))
