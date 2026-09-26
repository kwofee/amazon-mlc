import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.indexing import build_target_index
from business_entity_resolution.recall import evaluate_exact_rare_recall


class RecallTests(unittest.TestCase):
    def test_exact_and_rare_recall_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "source1.tsv"
            s2 = root / "s2.tsv"
            s3 = root / "s3.tsv"
            truth = root / "truth.tsv"
            folds = root / "folds.tsv"
            database = root / "targets.sqlite"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            source1.write_text(
                header
                + "S1-100\tJain Industries Pvt Ltd\tPlot B-78/1, Jaipur, PIN 302001\tIndia\n"
                + "S1-200\tNo Match Inc\t9 Empty Road\tUS\n",
                encoding="utf-8",
            )
            s2.write_text(
                header
                + "S2-1\tJain Industries Limited\tB-78/1, Jaipur, PIN 302001\tIndia\n"
                + "S2-2\tOther Jain Retail\tPlot 12, Delhi\tIndia\n",
                encoding="utf-8",
            )
            s3.write_text(
                header
                + "S3-1\tJain Industrial Services\tB-78/1 Jaipur\tIndia\n",
                encoding="utf-8",
            )
            truth.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-100\tS2-1\n"
                "S1-200\t\n",
                encoding="utf-8",
            )
            folds.write_text(
                "source1_entity_id\tfold\tcountry\tmatch_count\ts2_count\ts3_count\tsource_composition\tsingleton\n"
                "S1-100\t0\tIndia\t1\t1\t0\ts2_only\t0\n"
                "S1-200\t0\tUS\t0\t0\t0\tnone\t1\n",
                encoding="utf-8",
            )
            build_target_index([s2, s3], database)
            result = evaluate_exact_rare_recall(
                index_paths=[database],
                fold_path=folds,
                source1_path=source1,
                ground_truth_path=truth,
                fold=0,
                maximum_document_frequency=10,
                per_rare_route_limit=10,
                progress_every=0,
            )
            self.assertEqual(result["selection"]["anchor_count"], 2)
            self.assertEqual(result["recall"]["truth_pairs"], 1)
            self.assertEqual(result["recall"]["pair_recall"], 1.0)
            self.assertEqual(result["recall"]["complete_truth_anchor_recall"], 1.0)


    def test_bm25_and_numeric_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "source1.tsv"
            s2 = root / "s2.tsv"
            s3 = root / "s3.tsv"
            truth = root / "truth.tsv"
            folds = root / "folds.tsv"
            database = root / "targets.sqlite"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            source1.write_text(
                header
                + "S1-100\tJain Industries Pvt Ltd\tPlot B-78/1, MG Road, Jaipur, PIN 302001\tIndia\n"
                + "S1-200\tSmith Manufacturing Corp\t1234 Oak Avenue, Suite 500, Springfield, IL 62701\tUS\n",
                encoding="utf-8",
            )
            s2.write_text(
                header
                + "S2-1\tJain Industries Limited\tB-78/1, MG Road, Jaipur, PIN 302001\tIndia\n"
                + "S2-2\tSmith Mfg Corporation\t1234 Oak Ave, Ste 500, Springfield\tUS\n"
                + "S2-3\tUnrelated Business\t999 Elm St, Chicago\tUS\n",
                encoding="utf-8",
            )
            s3.write_text(
                header
                + "S3-1\tJain Industrial Works\tPlot 78, Jaipur\tIndia\n"
                + "S3-2\tSmith Manufacturing\t1234 Oak, Springfield, IL\tUS\n",
                encoding="utf-8",
            )
            truth.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-100\tS2-1,S3-1\n"
                "S1-200\tS2-2,S3-2\n",
                encoding="utf-8",
            )
            folds.write_text(
                "source1_entity_id\tfold\tcountry\tmatch_count\ts2_count\ts3_count\tsource_composition\tsingleton\n"
                "S1-100\t0\tIndia\t2\t1\t1\tboth\t0\n"
                "S1-200\t0\tUS\t2\t1\t1\tboth\t0\n",
                encoding="utf-8",
            )
            build_target_index([s2, s3], database)
            result = evaluate_exact_rare_recall(
                index_paths=[database],
                fold_path=folds,
                source1_path=source1,
                ground_truth_path=truth,
                fold=0,
                maximum_document_frequency=100,
                per_rare_route_limit=10,
                progress_every=0,
                enable_bm25=True,
                bm25_name_limit=5,
                bm25_address_limit=5,
                enable_numeric=True,
                numeric_limit=10,
            )
            self.assertEqual(result["selection"]["anchor_count"], 2)
            self.assertGreaterEqual(result["recall"]["pair_recall"], 0.5)
            self.assertIn("bm25_name", result["route_positive_hits"])


if __name__ == "__main__":
    unittest.main()
