# Phase 1 implementation plan: steps 1–6

Status: implementation started. This resource covers the first six milestones from the finalized Phase 1 plan in `PROJECT_MEMORY.md`.

## Scope

1. Exact macro entity-level F0.5 scorer.
2. Deterministic grouped S1-anchor validation folds.
3. Reusable name/address normalization and numeric/postal extraction.
4. Unit tests for normalization, compound numeric identifiers, scoring, folds, and indexes.
5. Disk-backed exact target lookup indexes.
6. Disk-backed rare-token inverted indexes with document frequencies.

This scope does not yet generate fuzzy BM25, character n-gram, embedding, or final model candidates. It establishes the representations, exact/rare candidate routes, and validation foundation those stages require.

## Design decisions

- All challenge files are read with `delimiter="\t"`.
- Source TSVs remain immutable.
- Derived data and indexes live outside `dataset/`.
- Country is treated as an arbitrary string.
- Target source comes from the `S2-`/`S3-` entity ID prefix.
- Validation groups are S1 anchors. Every labeled target for an anchor remains with that anchor.
- Fold assignment is a deterministic hash of seed, stratum, and S1 ID. The stratum contains country, truth-set size, and S2/S3 composition.
- Numeric extraction retains ordered tokens, a sorted multiset signature, and a set. Confident postal codes are stored separately.
- Exact and token indexes use SQLite so index size grows on disk rather than in Python dictionaries.
- Phase 1 has no transliteration component.

## Deliverables and acceptance criteria

### 1. Macro F0.5 scorer

Files: `src/business_entity_resolution/metrics.py`, CLI command `score`.

Acceptance criteria:

- Uses per-anchor set precision and recall.
- Correctly awards 1.0 to true singletons predicted as empty and 0.0 to false singleton matches.
- Macro-averages across every truth anchor.
- Rejects duplicate S1 rows, missing S1 rows, extra S1 rows, and malformed headers in strict CLI mode.
- Emits JSON containing macro score and anchor counts.
- Uses a temporary SQLite join for full-file scoring rather than loading millions of ID sets into RAM.

### 2. Grouped folds

Files: `src/business_entity_resolution/folds.py`, CLI command `make-folds`.

Acceptance criteria:

- Produces one row per training S1.
- Fold assignment is deterministic for a fixed seed and fold count.
- Output includes country, match count, S2 count, S3 count, source composition, singleton flag, and fold.
- Uses a temporary SQLite S1-country lookup rather than requiring all S1 IDs in memory.

### 3. Normalization and numeric/postal extraction

File: `src/business_entity_resolution/normalization.py`.

Acceptance criteria:

- Provides raw-preserving full/core/sorted name views.
- Provides full/useful address-token views.
- Preserves compound identifiers such as `B-78/1`, `G-3/571`, and `5A1`.
- Creates ordered and sorted numeric signatures with token delimiters.
- Separates high-confidence explicitly cued postal codes.
- Keeps ambiguous bare postal-looking values in the numeric signature.
- Treats literal `null` as a noise/missing token in derived address tokens.

### 4. Tests

Location: `tests/`.

Run:

```bash
cd code/business_entity_resolution
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Acceptance criterion: every test passes under the workspace Python version without network access.

### 5–6. Exact and rare-token indexes

File: `src/business_entity_resolution/indexing.py`, CLI command `build-index`.

Exact indexed views:

- normalized full name;
- ordered core name;
- sorted core-name multiset;
- domain stem;
- normalized full address;
- ordered numeric signature;
- sorted numeric signature;
- high-confidence postal code.

Token posting fields:

- core name;
- useful address;
- numeric/alphanumeric address.

Acceptance criteria:

- Index construction streams S2/S3 TSVs and commits batches.
- Postings are partitioned by country and source.
- Token document frequencies are materialized after loading.
- Query methods return exact bucket size and capped IDs.
- Rare-token queries rank candidates by rarest supporting token and number of supporting tokens.
- A small end-to-end fixture confirms S2/S3 and country isolation.

## Execution sequence

1. Run unit tests.
2. Run CLI help and small fixture smoke tests.
3. Generate full fold assignments under `artifacts/folds/`.
4. Inspect fold counts and strata before model work.
5. Build a small deterministic target-index shard and inspect bucket sizes.
6. Estimate full SQLite size and build time.
7. Build full training exact/rare index under `artifacts/indexes/` only after the shard check passes.
8. Measure exact/rare blocking recall as the next project milestone.

## Expected commands

```bash
cd code/business_entity_resolution

PYTHONPATH=src python3 -m business_entity_resolution.cli score \
  --truth ../../dataset/train/train_ground_truth.tsv \
  --predictions path/to/validation_predictions.tsv

PYTHONPATH=src python3 -m business_entity_resolution.cli make-folds \
  --source1 ../../dataset/train/train_source1.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --output artifacts/folds/train_folds.tsv \
  --folds 5 \
  --seed amazon-ml-2026-v1

PYTHONPATH=src python3 -m business_entity_resolution.cli build-index \
  --database artifacts/indexes/train_targets.sqlite \
  --targets ../../dataset/train/train_source2.tsv ../../dataset/train/train_source3.tsv
```

For a deterministic real-data shard before the full build:

```bash
PYTHONPATH=src python3 -m business_entity_resolution.cli build-index \
  --database artifacts/indexes/train_targets_10k_each.sqlite \
  --targets ../../dataset/train/train_source2.tsv ../../dataset/train/train_source3.tsv \
  --max-rows-per-file 10000
```

The full index build is intentionally not part of the unit-test command because it processes more than ten million target rows and may require substantial disk and wall-clock time.

After the full index exists, measure the current exact/rare blocker first on a deterministic one-percent sample of fold 0:

```bash
PYTHONPATH=src python3 -m business_entity_resolution.cli evaluate-recall \
  --index artifacts/indexes/train_s2_full.sqlite artifacts/indexes/train_s3_full.sqlite \
  --fold-file artifacts/folds/train_folds.tsv \
  --source1 ../../dataset/train/train_source1.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --fold 0 \
  --sample-modulo 100 \
  --sample-remainder 0
```
