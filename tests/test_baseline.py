import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from tests.support import REPO, pitch, run_step, sparse_outcome_rows, write_csv


PUBLIC_RESULTS = {
    "baseline_results.csv": "62468d69c347170fa4d8efbbbfd83d7ece1ccffc7650e5cf1a40c64096a36449",
    "lstm_weighted_macro_result.csv": "f29509374171fd672860d27bec4a561ba4abffc51431ba04e58357264055dc82",
    "outcome_empirical_eval.csv": "3d5b0267800bc56034d2f657b0e56a1220e8accd2bfb708d18e40a684215f314",
    "outcome_recommendation_sample.csv": "c6931fd84cb23c2b17f4c5db746245b7df41a69260dc85a96dac4abf3ac8759a",
    "transformer_weighted_macro_result.csv": "f1a7c056aa3f34bb4cb18d3c401632a70b62d14523f4b636488e1a18cd55c10a",
}


class BaselineTests(unittest.TestCase):
    def test_public_experiment_files_match_published_contents(self):
        for name, digest in PUBLIC_RESULTS.items():
            with self.subTest(name=name):
                data = (REPO / "outputs" / name).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_sequence_rare_pitch_boundary_and_labels_absent_from_training(self):
        # 70% of 1,500 targets is 1,050. SL and FF straddle the cutoff.
        targets = ["SL"] * 499 + ["FF"] * 500 + ["CH"] * 51 + ["KC"] * 450
        rows = [
            row
            for at_bat, target in enumerate(targets, 1)
            for row in (pitch(at_bat, 1, "SL"), pitch(at_bat, 2, target))
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/raw/savant_data.csv", rows)
            run_step(root, "preprocess")
            run_step(root, "make_sequence")
            result = pd.read_csv(root / "data/processed/sequence_data.csv")
            self.assertEqual(len(result), 1500)
            self.assertEqual(result["at_bat_number"].tolist(), list(range(1, 1501)))
            self.assertEqual(result["pitch_number"].unique().tolist(), [2])
            self.assertEqual(result["target_pitch_type_original"].tolist(), targets)
            self.assertEqual(result["target_pitch_type"].tolist(),
                             ["OTHER"] * 499 + ["FF"] * 500 + ["OTHER"] * 51 + ["KC"] * 450)
            self.assertTrue(result["seq_pitch_type"].eq("PAD|PAD|PAD|PAD|OTHER").all())

    def test_middle_pitch_with_missing_outcome_is_removed_only_from_outcome_history(self):
        rows = [pitch(1, number, "FF", zone=number) for number in range(1, 9)]
        rows[2]["delta_run_exp"] = None
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/raw/savant_data.csv", reversed(rows))
            write_csv(root, "data/raw/savant_data_with_outcome.csv", reversed(rows))
            for step in ("preprocess", "make_sequence", "make_outcome_sequence"):
                run_step(root, step)

            classification = pd.read_csv(root / "data/processed/sequence_data.csv").set_index("pitch_number")
            outcome = pd.read_csv(root / "data/processed/outcome_sequence_data.csv").set_index("pitch_number")
            self.assertEqual(classification.index.tolist(), [2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(outcome.index.tolist(), [2, 4, 5, 6, 7, 8])
            self.assertEqual(classification.loc[4, "seq_zone"], "ZPAD|ZPAD|Z1|Z2|Z3")
            self.assertEqual(outcome.loc[4, "seq_zone"], "ZPAD|ZPAD|ZPAD|Z1|Z2")
            self.assertEqual(classification.loc[8, "seq_zone"], "Z3|Z4|Z5|Z6|Z7")
            self.assertEqual(outcome.loc[8, "seq_zone"], "Z2|Z4|Z5|Z6|Z7")

    def test_offline_duplicate_prediction_missing_join_and_stable_ties(self):
        predictions = [
            {"game_pk": 100, "at_bat_number": 13, "pitch_number": 2, "top3_pitch_type": value}
            for value in ("SL|FF", "FF|SL")
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/processed/outcome_sequence_data.csv", sparse_outcome_rows())
            write_csv(root, "outputs/lstm_weighted_macro_predictions.csv", predictions)
            run_step(root, "outcome_empirical_recommender")
            result = pd.read_csv(root / "outputs/outcome_empirical_recommendations.csv")
            self.assertEqual(result["at_bat_number"].tolist(), [13, 14, 15])
            self.assertEqual(result.loc[0, "lstm_top3_pitch_type"], "SL|FF")
            self.assertTrue(result.loc[1:, "lstm_top3_pitch_type"].isna().all())
            self.assertEqual(result["recommended_pitch_type"].tolist(), ["SL", "FF", "FF"])
            self.assertEqual(result["recommended_zone"].tolist(), ["Z1", "Z1", "Z1"])
            self.assertEqual(result["recommendation_source"].tolist(), ["global"] * 3)
            self.assertEqual(result["recommendation_source_count"].tolist(), [12] * 3)
            self.assertEqual(
                result.loc[0, ["rec_1", "rec_2", "rec_3", "rec_4"]].tolist(),
                ["SL/Z1:0.1000(global,n=12)", "SL/Z2:0.1000(global,n=12)",
                 "FF/Z1:0.1000(global,n=12)", "FF/Z2:0.1000(global,n=12)"],
            )
