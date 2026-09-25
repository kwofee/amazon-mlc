# %% [markdown]
# # Ground-truth address matches
#
# Load the first 10,000 ground-truth rows and retrieve the raw matching
# addresses from Sources 1, 2, and 3.

# %%
from pathlib import Path

import pandas as pd


train_dir = Path("dataset/train")

ground_truth = pd.read_csv(
    train_dir / "train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=10_000,
)

print(f"Loaded {len(ground_truth):,} ground-truth rows")
ground_truth.head()

# %%
pairs = ground_truth.reset_index(names="ground_truth_index")
pairs["ground_truth_row"] = pairs["ground_truth_index"] + 1
pairs["matched_entity_id"] = pairs["matched_entity_ids"].str.split(",")
pairs = pairs.explode("matched_entity_id")
pairs["matched_entity_id"] = pairs["matched_entity_id"].str.strip()
pairs = pairs[pairs["matched_entity_id"] != ""].copy()

source1_ids = set(ground_truth["source1_entity_id"])
source2_ids = set(
    pairs.loc[
        pairs["matched_entity_id"].str.startswith("S2-"),
        "matched_entity_id",
    ]
)
source3_ids = set(
    pairs.loc[
        pairs["matched_entity_id"].str.startswith("S3-"),
        "matched_entity_id",
    ]
)


def read_selected_records(path, wanted_ids):
    selected_chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    ):
        selected = chunk[chunk["entity_id"].isin(wanted_ids)]

        if not selected.empty:
            selected_chunks.append(selected)

    return pd.concat(selected_chunks, ignore_index=True)


source1 = read_selected_records(
    train_dir / "train_source1.tsv",
    source1_ids,
)
source2 = read_selected_records(
    train_dir / "train_source2.tsv",
    source2_ids,
)
source3 = read_selected_records(
    train_dir / "train_source3.tsv",
    source3_ids,
)

# %%
source1_addresses = source1[
    ["entity_id", "business_address", "country"]
].rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_address": "source1_address",
        "country": "source1_country",
    }
)

matched_addresses = pd.concat(
    [source2, source3],
    ignore_index=True,
)[["entity_id", "business_address", "country"]].rename(
    columns={
        "entity_id": "matched_entity_id",
        "business_address": "matched_address",
        "country": "matched_country",
    }
)

address_matches = (
    pairs[
        ["ground_truth_row", "source1_entity_id", "matched_entity_id"]
    ]
    .merge(source1_addresses, on="source1_entity_id", how="left")
    .merge(matched_addresses, on="matched_entity_id", how="left")
)

address_matches = address_matches[
    [
        "ground_truth_row",
        "source1_entity_id",
        "source1_address",
        "source1_country",
        "matched_entity_id",
        "matched_address",
        "matched_country",
    ]
]

print(f"Created {len(address_matches):,} matched address rows")
address_matches.head(100)
