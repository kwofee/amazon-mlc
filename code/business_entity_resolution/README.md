# Business Entity Resolution Pipeline

Amazon ML Challenge 2026 — entity resolution across three independent business record sources.

## Problem

Given a deduplicated reference source (S1, ~2.2M anchors), identify matching records in two noisy target sources (S2, ~5M records; S3, ~5.3M records). The leaderboard metric is macro entity-level F-beta with beta=0.5 (precision-weighted).

## Current Implementation Status

**Blocking stage: complete.** Achieves 94.6% pair recall with 288 mean candidates per anchor.

### Implemented Components

1. **Multi-view normalization** (`normalization.py`)
   - NFKC + casefolding + legal-token stripping
   - Full name, core name (legal suffixes removed), sorted core name
   - Address canonical tokens, useful address tokens, numeric signatures
   - Postal code extraction with confidence scoring
   - Devanagari-aware tokenization (preserves combining marks)
   - Domain/URL stem extraction for website-style names

2. **SQLite inverted indexes** (`indexing.py`)
   - Exact posting index: full name, core name, sorted name, domain, address, numeric signatures, postal codes
   - Token postings with document frequency stats for rare-token retrieval
   - BM25 scoring over token postings (configurable max_token_df threshold)
   - Numeric overlap retrieval (shared address numbers + locality tokens)
   - Character n-gram index (4-5 grams, two-pass build: count DFs, store only rare n-grams)

3. **Candidate recall evaluation** (`recall.py`)
   - 5 retrieval routes: exact match, rare token, BM25, numeric overlap, character n-gram
   - Per-route unique hit attribution
   - Country and source slice metrics
   - Grouped anchor fold evaluation

4. **Validation infrastructure**
   - Macro entity-level F0.5 scorer (`metrics.py`)
   - Grouped anchor folds preventing label leakage (`folds.py`)
   - 18 unit tests covering all components

### Blocking Recall Results

Evaluated on fold 0, 913 sampled anchors, full S2+S3 indexes (10.3M targets):

| Metric | Value |
|---|---:|
| Pair recall | 94.6% |
| Any-hit anchor recall | 99.2% |
| Complete-set anchor recall | 85.0% |
| Mean candidates/anchor | 288 |

Route unique positive hits: n-gram (53), rare_numeric (27), bm25_address (26), rare_address (19).

Country recall: US 95.4%, India 93.5%.

## Environment

Python 3.9+. Uses only the Python standard library (no external dependencies).

## Quick Start

```bash
# Set up bytecode cache (macOS managed Python may deny default cache writes)
export PYTHONPYCACHEPREFIX=/tmp/ber_pycache
export PYTHONPATH=src

# Run tests
python3 -m unittest discover -s tests -v

# Generate grouped validation folds
python3 -m business_entity_resolution.cli make-folds \
  --source1 ../../dataset/train/train_source1.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --output artifacts/folds/train_folds.tsv

# Build target indexes (takes ~1hr per source on 8-core machine)
python3 -m business_entity_resolution.cli build-index \
  --database artifacts/indexes/train_s2_full.sqlite \
  --targets ../../dataset/train/train_source2.tsv \
  --skip-stats

python3 -m business_entity_resolution.cli materialize-stats \
  --database artifacts/indexes/train_s2_full.sqlite

# Build character n-gram indexes (~16 min per source)
python3 -m business_entity_resolution.cli build-ngram-index \
  --database artifacts/indexes/train_s2_full.sqlite \
  --targets ../../dataset/train/train_source2.tsv \
  --max-ngram-df 1000

# Evaluate blocking recall
python3 -m business_entity_resolution.cli evaluate-recall \
  --index artifacts/indexes/train_s2_full.sqlite artifacts/indexes/train_s3_full.sqlite \
  --fold-file artifacts/folds/train_folds.tsv \
  --source1 ../../dataset/train/train_source1.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --fold 0 --sample-modulo 500 \
  --enable-bm25 --bm25-max-token-df 5000 \
  --enable-numeric \
  --enable-ngram
```

## CLI Commands

| Command | Purpose |
|---|---|
| `score` | Compute exact macro entity F-beta |
| `make-folds` | Generate grouped anchor validation folds |
| `build-index` | Build exact + rare-token SQLite indexes |
| `materialize-stats` | Materialize token_stats after `--skip-stats` build |
| `build-ngram-index` | Build character n-gram posting index |
| `evaluate-recall` | Measure blocker recall on held-out anchor folds |

## Project Structure

```
src/business_entity_resolution/
  cli.py             # CLI entry points
  normalization.py   # Multi-view name/address normalization
  indexing.py        # SQLite index building and retrieval
  recall.py          # Candidate recall evaluation
  metrics.py         # Macro entity-level F0.5 scoring
  folds.py           # Grouped anchor fold generation
  io_utils.py        # TSV streaming utilities

tests/
  test_normalization.py
  test_indexing.py
  test_recall.py
  test_metrics.py
  test_folds.py

artifacts/
  indexes/           # Generated SQLite index files (gitignored, multi-GB)
  folds/             # Generated fold assignments (gitignored)
```

## Next Steps

1. Pairwise feature extraction (name/address similarity, numeric agreement, route/rank features)
2. Hard negative mining from blocker candidate pools
3. Gradient-boosted tree matcher training
4. Source-specific calibration (S2 vs S3 corruption differs)
5. Global decoder with target-to-anchor exclusivity
6. Cross-script/transliteration retrieval for Devanagari targets
7. France zero-shot transfer

See `PROJECT_MEMORY.md` at the repository root for detailed analysis, methodology, and experiment results.
