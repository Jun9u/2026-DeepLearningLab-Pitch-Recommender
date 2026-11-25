import json
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import outcome_empirical_recommender as recommender
from src.empirical_ranker import select_candidate_pitches, top_pitch_types
import pandas as pd
from tests.support import REPO, command, sparse_outcome_rows, write_csv


class RankerDiagnosticsTests(unittest.TestCase):
    script = str(REPO / "src/outcome_empirical_recommender.py")

    def test_candidate_rules_can_exhaust_all_pitch_sources(self):
        self.assertEqual(select_candidate_pitches(("OTHER", ""), (" None ",), ("nan",), {"FF"}), [])
        pitches = top_pitch_types(pd.Series(["OTHER"] * 4 + ["FF"] * 3 + ["SL"] * 2 + ["CH"]))
        self.assertEqual(pitches, ["FF", "SL", "CH"])
        self.assertEqual(select_candidate_pitches(("UN", " ch "), pitches, (), set(pitches)), ["CH", "FF", "SL"])

    def test_compares_models_on_the_same_test_rows_without_changing_csv_outputs(self):
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
            self.assertEqual(report["result_sha256"], {path.name: sha256(blob).hexdigest()
                             for path, blob in zip(results, originals, strict=True)})
            command(root, self.script)
            self.assertFalse(diagnostics.exists())

        self.assertEqual((report["train_rows"], report["test_rows"]), (16, 4))
        self.assertAlmostEqual(report["train_global_mean"], 0.2)
        metrics = report["observed_action_metrics"]
        self.assertAlmostEqual(metrics["train_global_mean"]["mae"], 0.15)
        self.assertAlmostEqual(metrics["empirical_ranker"]["mae"], 10 / 58 - 0.05)
        self.assertAlmostEqual(metrics["train_pitch_zone_mean"]["mae"], 0.05)
        self.assertEqual(report["lstm_top3_join"], {"matched_rows": 1, "missing_rows": 3, "match_rate": 0.25})
        self.assertEqual(report["observed_action_source_counts"], {"pitcher+count+action+zone": 4})

    def test_pitch_zone_mean_and_candidate_coverage_on_held_out_games(self):
        games = [
            {"game_date": f"2025-06-0{offset}", "game_pk": offset, "split": split}
            for offset, split in enumerate(("train", "validation", "test"), start=1)
        ]
        actions = [
            (1, 1, "FF", "Z1", 0.0),
            (2, 2, "SL", "Z2", 0.4),
            (3, 1, "FF", "Z1", 0.0),
            (3, 1, "SL", "Z2", 0.4),
            (3, 1, "FF", "Z2", 0.3),
            (3, 1, "FF", "Z9", 0.2),
            (3, 3, "SL", "Z2", 0.2),
            (3, 3, "OTHER", "Z2", 0.2),
        ]
        rows = [
            {
                "game_date": games[game - 1]["game_date"], "game_pk": game,
                "at_bat_number": at_bat, "pitch_number": 2, "pitcher": pitcher, "balls": 1, "strikes": 1,
                "action_pitch_type": pitch, "action_zone": zone, "delta_run_exp": value,
            }
            for at_bat, (game, pitcher, pitch, zone, value) in enumerate(actions, start=1)
        ]
        predictions = [
            {"game_pk": 3, "at_bat_number": 4, "pitch_number": 2, "top3_pitch_type": "OTHER|SL|SL"},
            {"game_pk": 3, "at_bat_number": 8, "pitch_number": 2, "top3_pitch_type": "OTHER"},
        ]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "outcomes.csv", rows)
            write_csv(root, "game_split.csv", games)
            split_hash = sha256((root / "game_split.csv").read_bytes()).hexdigest()
            write_csv(root, "predictions.csv", [{**row, "game_split_sha256": split_hash} for row in predictions])
            diagnostics = root / "diagnostics.json"
            with patch.dict(os.environ, {"PITCH_GAME_SPLIT": str(root / "game_split.csv")}), patch.multiple(
                recommender, DATA_PATH=root / "outcomes.csv", LSTM_PRED_PATH=root / "predictions.csv",
                EVAL_OUTPUT_PATH=root / "eval.csv", RECOMMEND_OUTPUT_PATH=root / "recommendations.csv",
                DIAGNOSTICS_OUTPUT_PATH=diagnostics,
            ):
                recommender.main(diagnostics=True)
                previous = (root / "eval.csv").read_bytes()
                write_csv(root, "predictions.csv", [{**row, "game_split_sha256": "different"} for row in predictions])
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    recommender.main(diagnostics=True)
                write_csv(root, "predictions.csv", predictions)
                with self.assertRaisesRegex(ValueError, "경기 분할 정보"):
                    recommender.main(diagnostics=True)
                self.assertEqual((root / "eval.csv").read_bytes(), previous)
            report = json.loads(diagnostics.read_text(encoding="utf-8"))

        self.assertEqual((report["train_rows"], report["test_rows"]), (2, 6))
        self.assertAlmostEqual(report["observed_action_metrics"]["train_pitch_zone_mean"]["mae"], 0.05)
        self.assertEqual(report["train_pitch_zone_mean_counts"], {"pair_mean_rows": 3, "global_fallback_rows": 3})
        self.assertEqual(report["candidate_coverage"], {
            "lstm_top3_with_frequency": {
                "rows_with_candidates": 6, "observed_action_covered_rows": 4,
                "observed_action_coverage_rate": 4 / 6, "average_candidates_per_row": 3.0,
            },
            "frequency_only": {
                "rows_with_candidates": 6, "observed_action_covered_rows": 3,
                "observed_action_coverage_rate": 0.5, "average_candidates_per_row": 16 / 6,
            },
        })

    def test_reports_zero_matches_and_null_r2_for_one_test_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/processed/outcome_sequence_data.csv", sparse_outcome_rows()[:5])
            command(root, self.script, "--diagnostics")
            report = json.loads((root / "results/outcome_empirical_diagnostics.json").read_text(encoding="utf-8"))

        self.assertEqual(report["lstm_top3_join"], {"matched_rows": 0, "missing_rows": 1, "match_rate": 0.0})
        for metrics in report["observed_action_metrics"].values():
            self.assertIsNone(metrics["r2"])

    def test_constant_targets_use_null_r2_for_matching_and_different_scores(self):
        aligned = recommender.regression_metrics([0.2, 0.2], [0.2, 0.2])
        self.assertTrue(pd.isna(aligned["r2"]))
        self.assertEqual(aligned["mae"], 0)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = sparse_outcome_rows()[:10]
            for row in rows[-2:]:
                row["delta_run_exp"] = 0.2
            write_csv(root, "data/processed/outcome_sequence_data.csv", rows)
            command(root, self.script, "--diagnostics")
            report = json.loads((root / "results/outcome_empirical_diagnostics.json").read_text(encoding="utf-8"))
            evaluation = pd.read_csv(root / "results/outcome_empirical_eval.csv")

        self.assertEqual(report["test_rows"], 2)
        self.assertTrue(pd.isna(evaluation.iloc[0]["r2"]))
        for metrics in report["observed_action_metrics"].values():
            self.assertIsNone(metrics["r2"])
            self.assertGreater(metrics["mae"], 0)
