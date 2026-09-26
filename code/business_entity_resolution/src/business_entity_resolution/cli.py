"""Command-line entry points for the Phase 1 foundation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .folds import generate_grouped_folds
from .indexing import build_ngram_index, build_target_index, populate_raw_records
from .matching import generate_training_pairs, train_matcher
from .metrics import score_files
from .recall import evaluate_exact_rare_recall


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="compute exact macro entity F-beta")
    score.add_argument("--truth", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--beta", type=float, default=0.5)

    folds = subparsers.add_parser("make-folds", help="generate grouped anchor folds")
    folds.add_argument("--source1", required=True)
    folds.add_argument("--ground-truth", required=True)
    folds.add_argument("--output", required=True)
    folds.add_argument("--folds", type=int, default=5)
    folds.add_argument("--seed", default="amazon-ml-2026-v1")

    index = subparsers.add_parser(
        "build-index", help="build exact and rare-token SQLite indexes"
    )
    index.add_argument("--database", required=True)
    index.add_argument("--targets", required=True, nargs="+")
    index.add_argument("--batch-postings", type=int, default=100_000)
    index.add_argument("--overwrite", action="store_true")
    index.add_argument("--resume", action="store_true")
    index.add_argument(
        "--max-rows-per-file",
        type=int,
        help="optional deterministic prefix size for shard/smoke builds",
    )
    index.add_argument(
        "--skip-stats",
        action="store_true",
        help="skip token_stats materialization (run materialize-stats separately)",
    )

    stats = subparsers.add_parser(
        "materialize-stats",
        help="materialize token_stats from token_postings (run after build-index --skip-stats)",
    )
    stats.add_argument("--database", required=True)

    recall = subparsers.add_parser(
        "evaluate-recall", help="measure exact/rare blocker recall on an anchor fold"
    )
    recall.add_argument("--index", required=True, nargs="+")
    recall.add_argument("--fold-file", required=True)
    recall.add_argument("--source1", required=True)
    recall.add_argument("--ground-truth", required=True)
    recall.add_argument("--fold", type=int, default=0)
    recall.add_argument("--sample-modulo", type=int)
    recall.add_argument("--sample-remainder", type=int, default=0)
    recall.add_argument("--max-document-frequency", type=int, default=5_000)
    recall.add_argument("--rare-route-limit", type=int, default=50)
    recall.add_argument("--exact-bucket-cap", type=int, default=500)
    recall.add_argument("--progress-every", type=int, default=1_000)
    recall.add_argument("--enable-bm25", action="store_true")
    recall.add_argument("--bm25-name-limit", type=int, default=20)
    recall.add_argument("--bm25-address-limit", type=int, default=10)
    recall.add_argument("--bm25-max-token-df", type=int, default=5_000)
    recall.add_argument("--enable-numeric", action="store_true")
    recall.add_argument("--numeric-limit", type=int, default=50)
    recall.add_argument("--enable-ngram", action="store_true")
    recall.add_argument("--ngram-name-limit", type=int, default=20)

    ngram = subparsers.add_parser(
        "build-ngram-index", help="build character n-gram posting index"
    )
    ngram.add_argument("--database", required=True)
    ngram.add_argument("--targets", required=True, nargs="+")
    ngram.add_argument("--max-ngram-df", type=int, default=1000)
    ngram.add_argument("--ngram-min", type=int, default=4)
    ngram.add_argument("--ngram-max", type=int, default=5)
    ngram.add_argument(
        "--max-rows-per-file", type=int,
        help="optional prefix size for shard builds",
    )
    populate = subparsers.add_parser(
        "populate-raw-records",
        help="store raw target records in the SQLite index for fast lookup",
    )
    populate.add_argument("--database", required=True)
    populate.add_argument("--targets", required=True, nargs="+")

    pairs = subparsers.add_parser(
        "generate-training-pairs",
        help="generate labeled training pairs from blocker candidates",
    )
    pairs.add_argument("--index", required=True, nargs="+")
    pairs.add_argument("--fold-file", required=True)
    pairs.add_argument("--source1", required=True)
    pairs.add_argument("--ground-truth", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--fold", type=int, default=0)
    pairs.add_argument("--sample-modulo", type=int)
    pairs.add_argument("--sample-remainder", type=int, default=0)
    pairs.add_argument("--negatives-per-anchor", type=int, default=20)
    pairs.add_argument("--max-document-frequency", type=int, default=5_000)
    pairs.add_argument("--bm25-max-token-df", type=int, default=5_000)
    pairs.add_argument("--progress-every", type=int, default=500)

    train = subparsers.add_parser(
        "train-matcher",
        help="train HistGradientBoostingClassifier on generated pairs",
    )
    train.add_argument("--train-pairs", required=True)
    train.add_argument("--model-output", required=True)
    train.add_argument("--val-pairs")
    train.add_argument("--max-iter", type=int, default=500)
    train.add_argument("--learning-rate", type=float, default=0.1)
    train.add_argument("--max-depth", type=int, default=6)
    train.add_argument("--min-samples-leaf", type=int, default=50)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = _parser().parse_args(argv)
    if arguments.command == "score":
        result = score_files(arguments.truth, arguments.predictions, beta=arguments.beta)
    elif arguments.command == "make-folds":
        result = generate_grouped_folds(
            source1_path=arguments.source1,
            ground_truth_path=arguments.ground_truth,
            output_path=arguments.output,
            fold_count=arguments.folds,
            seed=arguments.seed,
        )
    elif arguments.command == "build-index":
        result = build_target_index(
            target_paths=arguments.targets,
            database_path=arguments.database,
            batch_posting_count=arguments.batch_postings,
            overwrite=arguments.overwrite,
            resume=arguments.resume,
            maximum_rows_per_file=arguments.max_rows_per_file,
            skip_stats=arguments.skip_stats,
        )
    elif arguments.command == "materialize-stats":
        import sqlite3 as _sqlite3
        db = Path(arguments.database)
        conn = _sqlite3.connect(db)
        conn.execute("PRAGMA cache_size=-1048576")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("DELETE FROM token_stats")
        conn.execute(
            "INSERT INTO token_stats "
            "SELECT country, source, field, token, COUNT(*) "
            "FROM token_postings GROUP BY country, source, field, token"
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM token_stats").fetchone()[0]
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.close()
        result = {"token_stats_rows": count}
    elif arguments.command == "evaluate-recall":
        result = evaluate_exact_rare_recall(
            index_paths=arguments.index,
            fold_path=arguments.fold_file,
            source1_path=arguments.source1,
            ground_truth_path=arguments.ground_truth,
            fold=arguments.fold,
            sample_modulo=arguments.sample_modulo,
            sample_remainder=arguments.sample_remainder,
            maximum_document_frequency=arguments.max_document_frequency,
            per_rare_route_limit=arguments.rare_route_limit,
            exact_bucket_cap=arguments.exact_bucket_cap,
            progress_every=arguments.progress_every,
            enable_bm25=arguments.enable_bm25,
            bm25_name_limit=arguments.bm25_name_limit,
            bm25_address_limit=arguments.bm25_address_limit,
            bm25_max_token_df=arguments.bm25_max_token_df,
            enable_numeric=arguments.enable_numeric,
            numeric_limit=arguments.numeric_limit,
            enable_ngram=arguments.enable_ngram,
            ngram_name_limit=arguments.ngram_name_limit,
        )
    elif arguments.command == "build-ngram-index":
        result = build_ngram_index(
            target_paths=arguments.targets,
            database_path=arguments.database,
            max_ngram_df=arguments.max_ngram_df,
            ngram_min=arguments.ngram_min,
            ngram_max=arguments.ngram_max,
            maximum_rows_per_file=arguments.max_rows_per_file,
        )
    elif arguments.command == "populate-raw-records":
        result = populate_raw_records(
            database_path=arguments.database,
            target_paths=arguments.targets,
        )
    elif arguments.command == "generate-training-pairs":
        result = generate_training_pairs(
            index_paths=arguments.index,
            fold_path=arguments.fold_file,
            source1_path=arguments.source1,
            ground_truth_path=arguments.ground_truth,
            output_path=arguments.output,
            fold=arguments.fold,
            sample_modulo=arguments.sample_modulo,
            sample_remainder=arguments.sample_remainder,
            negatives_per_anchor=arguments.negatives_per_anchor,
            max_df=arguments.max_document_frequency,
            bm25_max_token_df=arguments.bm25_max_token_df,
            progress_every=arguments.progress_every,
        )
    elif arguments.command == "train-matcher":
        result = train_matcher(
            train_pairs_path=arguments.train_pairs,
            model_output_path=arguments.model_output,
            val_pairs_path=arguments.val_pairs,
            max_iter=arguments.max_iter,
            learning_rate=arguments.learning_rate,
            max_depth=arguments.max_depth,
            min_samples_leaf=arguments.min_samples_leaf,
        )
    else:
        raise AssertionError(f"unhandled command: {arguments.command}")
    print(json.dumps(result.to_dict() if hasattr(result, "to_dict") else result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
