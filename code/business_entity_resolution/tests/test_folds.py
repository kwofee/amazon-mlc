import csv
import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.folds import generate_grouped_folds, stable_fold


class FoldTests(unittest.TestCase):
    def test_stable_fold(self):
        first = stable_fold("S1-1", "India|2|both", 5, "seed")
        second = stable_fold("S1-1", "India|2|both", 5, "seed")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 5)

    def test_generate_grouped_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source1.tsv"
            truth = root / "truth.tsv"
            output = root / "folds.tsv"
            source.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAlpha\tOne Road\tUS\n"
                "S1-2\tBeta\tTwo Road\tIndia\n",
                encoding="utf-8",
            )
            truth.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-2\tS2-2,S3-2\n"
                "S1-1\t\n",
                encoding="utf-8",
            )
            summary = generate_grouped_folds(source, truth, output, fold_count=3)
            self.assertEqual(summary["source1_count"], 2)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 2)
            india = next(row for row in rows if row["source1_entity_id"] == "S1-2")
            self.assertEqual(india["country"], "India")
            self.assertEqual(india["source_composition"], "both")
            self.assertEqual(india["match_count"], "2")


if __name__ == "__main__":
    unittest.main()
