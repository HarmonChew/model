"""Checks for the portfolio showcase results table."""

import csv
import unittest
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "docs" / "stage_results.csv"
COLUMNS = ["stage", "title", "corpus", "metric", "best_val_loss", "kept", "note", "source"]


class ShowcaseResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CSV_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cls.columns = reader.fieldnames
            cls.rows = list(reader)
        cls.by_stage = {int(r["stage"]): r for r in cls.rows}

    def test_columns_and_row_count(self):
        self.assertEqual(self.columns, COLUMNS)
        self.assertEqual(len(self.rows), 59)

    def test_stage_numbers_unique_and_ascending(self):
        stages = [int(r["stage"]) for r in self.rows]
        self.assertEqual(stages, sorted(set(stages)))
        self.assertEqual(stages, list(range(1, 60)))

    def test_corpus_switch_at_stage_37(self):
        for stage, row in self.by_stage.items():
            self.assertEqual(row["corpus"], "tiny" if stage <= 36 else "complete", stage)

    def test_losses_positive_or_blank(self):
        for stage, row in self.by_stage.items():
            if row["best_val_loss"]:
                self.assertGreater(float(row["best_val_loss"]), 0.0, stage)

    def test_kept_and_note(self):
        for stage, row in self.by_stage.items():
            self.assertIn(row["kept"], {"yes", "no"}, stage)
            self.assertLessEqual(len(row["note"].split()), 12, stage)

    def test_anchors(self):
        self.assertEqual(round(float(self.by_stage[59]["best_val_loss"]), 4), 1.4431)
        champion = float(self.by_stage[53]["best_val_loss"])
        self.assertTrue(1.3450 <= champion < 1.3460)  # 1.345x


if __name__ == "__main__":
    unittest.main()
