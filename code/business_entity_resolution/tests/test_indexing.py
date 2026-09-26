import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.indexing import TargetIndex, build_ngram_index, build_target_index


class IndexTests(unittest.TestCase):
    def test_exact_and_rare_token_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s2 = root / "s2.tsv"
            s3 = root / "s3.tsv"
            database = root / "targets.sqlite"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            s2.write_text(
                header
                + "S2-1\tJain Industries Pvt Ltd\tPlot B-78/1, Jaipur, PIN 302001\tIndia\n"
                + "S2-2\tJain Retail Pvt Ltd\tPlot 12, Delhi, PIN 110001\tIndia\n"
                + "S2-3\tJain Industries Inc\t12 Main Street, Austin, ZIP 78701\tUS\n",
                encoding="utf-8",
            )
            s3.write_text(
                header
                + "S3-1\tJain Industries Limited\tB-78/1, Jaipur, PIN 302001\tIndia\n",
                encoding="utf-8",
            )
            summary = build_target_index([s2, s3], database, batch_posting_count=10)
            self.assertEqual(summary["target_rows"], 4)

            with TargetIndex(database) as index:
                exact_s2 = index.exact_lookup(
                    "India", "S2", "name_core", "jain industries"
                )
                self.assertEqual(exact_s2.entity_ids, ("S2-1",))
                exact_s3 = index.exact_lookup(
                    "India", "S3", "name_core", "jain industries"
                )
                self.assertEqual(exact_s3.entity_ids, ("S3-1",))
                us = index.exact_lookup("US", "S2", "name_core", "jain industries")
                self.assertEqual(us.entity_ids, ("S2-3",))

                rare = index.rare_token_candidates(
                    "India",
                    "S2",
                    "name_core",
                    ["jain", "industries"],
                    maximum_document_frequency=2,
                )
                self.assertEqual(rare[0].entity_id, "S2-1")
                self.assertEqual(rare[0].supporting_token_count, 2)


    def test_ngram_index_and_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s2 = root / "s2.tsv"
            database = root / "targets.sqlite"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            s2.write_text(
                header
                + "S2-1\tAlpha Technologies Pvt Ltd\tPlot 12, Delhi\tIndia\n"
                + "S2-2\tBeta Solutions Inc\t100 Main St\tUS\n"
                + "S2-3\tAlpha Tech Limited\tPlot 12, Delhi\tIndia\n",
                encoding="utf-8",
            )
            build_target_index([s2], database, batch_posting_count=10)
            result = build_ngram_index(
                [s2], database, max_ngram_df=10, ngram_min=4, ngram_max=5
            )
            self.assertEqual(result["target_rows"], 3)
            self.assertGreater(result["eligible_ngrams"], 0)
            self.assertGreater(result["ngram_postings"], 0)

            with TargetIndex(database) as index:
                candidates = index.ngram_candidates(
                    "India", "S2", "name",
                    "Alpha Technologies Private Limited",
                    limit=5, ngram_min=4, ngram_max=5,
                )
                eids = {c.entity_id for c in candidates}
                self.assertIn("S2-1", eids)


if __name__ == "__main__":
    unittest.main()
