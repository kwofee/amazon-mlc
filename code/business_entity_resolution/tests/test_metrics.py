import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.metrics import entity_fbeta, macro_entity_fbeta, score_files


class MetricTests(unittest.TestCase):
    def test_entity_example_from_readme(self):
        score = entity_fbeta({"S2-1", "S3-1"}, {"S2-1", "S2-2", "S3-1"})
        self.assertAlmostEqual(score, 5 / 7)

    def test_singletons(self):
        self.assertEqual(entity_fbeta(set(), set()), 1.0)
        self.assertEqual(entity_fbeta(set(), {"S2-1"}), 0.0)

    def test_macro_is_per_anchor(self):
        result = macro_entity_fbeta(
            {"S1-1": set(), "S1-2": {"S2-1"}},
            {"S1-1": set(), "S1-2": set()},
        )
        self.assertEqual(result.macro_fbeta, 0.5)
        self.assertEqual(result.correctly_empty_singletons, 1)

    def test_strict_anchor_set(self):
        with self.assertRaises(ValueError):
            macro_entity_fbeta({"S1-1": set()}, {})

    def test_disk_backed_file_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.tsv"
            prediction = root / "prediction.tsv"
            truth.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-1\tS2-1,S3-1\n"
                "S1-2\t\n",
                encoding="utf-8",
            )
            prediction.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-2\t\n"
                "S1-1\tS2-1,S2-9,S3-1\n",
                encoding="utf-8",
            )
            result = score_files(truth, prediction)
            self.assertAlmostEqual(result.macro_fbeta, (1.0 + 5 / 7) / 2)


if __name__ == "__main__":
    unittest.main()
