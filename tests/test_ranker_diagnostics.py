import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.support import REPO, command, sparse_outcome_rows, write_csv


class RankerDiagnosticsTests(unittest.TestCase):
    script = str(REPO / "src/outcome_empirical_recommender.py")

    def test_compares_both_models_on_the_same_test_rows(self):
        rows = [
            {
                "game_date": "2025-06-01",
                "game_pk": 100,
                "at_bat_number": index + 1,
                "pitch_number": 2,
                "pitcher": 900002 if 8 <= index < 16 else 900001,
                "balls": 1,
                "strikes": 1,
                "action_pitch_type": "SL" if 8 <= index < 16 else "FF",
                "action_zone": "Z1",
                "delta_run_exp": value,
            }
            for index, value in enumerate([0.0] * 8 + [0.4] * 8 + [0.0, 0.1, 0.0, 0.1])
        ]
        predictions = [{"game_pk": 100, "at_bat_number": 17, "pitch_number": 2, "top3_pitch_type": "FF|SL"}]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/processed/outcome_sequence_data.csv", rows)
            write_csv(root, "results/lstm_weighted_macro_predictions.csv", predictions)
            command(root, self.script)
            output = root / "results"
            diagnostics = output / "outcome_empirical_diagnostics.json"
            self.assertFalse(diagnostics.exists())
            results = [output / f"outcome_empirical_{suffix}.csv" for suffix in ("eval", "recommendations")]
            originals = [result.read_bytes() for result in results]

            command(root, self.script, "--diagnostics")
            report = json.loads(diagnostics.read_text(encoding="utf-8"))
            self.assertEqual([result.read_bytes() for result in results], originals)

        self.assertEqual((report["train_rows"], report["test_rows"]), (16, 4))
        self.assertAlmostEqual(report["train_global_mean"], 0.2)
        metrics = report["observed_action_metrics"]
        self.assertAlmostEqual(metrics["train_global_mean"]["mae"], 0.15)
        self.assertAlmostEqual(metrics["empirical_ranker"]["mae"], 10 / 58 - 0.05)
        self.assertEqual(report["lstm_top3_join"], {"matched_rows": 1, "missing_rows": 3, "match_rate": 0.25})
        self.assertEqual(report["observed_action_source_counts"], {"pitcher+count+action+zone": 4})

    def test_reports_zero_matches_and_null_r2_for_one_test_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/processed/outcome_sequence_data.csv", sparse_outcome_rows()[:5])
            command(root, self.script, "--diagnostics")
            report = json.loads((root / "results/outcome_empirical_diagnostics.json").read_text(encoding="utf-8"))

        self.assertEqual(report["lstm_top3_join"], {"matched_rows": 0, "missing_rows": 1, "match_rate": 0.0})
        for metrics in report["observed_action_metrics"].values():
            self.assertIsNone(metrics["r2"])
