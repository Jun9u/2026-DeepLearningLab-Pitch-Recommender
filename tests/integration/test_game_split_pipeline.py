from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from tests.support import pipeline_pitches, run_step, write_csv


class GameSplitPipelineTests(unittest.TestCase):
    def test_training_evaluation_and_export_use_the_same_game_assignments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = pipeline_pitches()
            # Introduce a target that neither classifier sees during training.
            for row in rows:
                if row["game_pk"] in {106, 108} and row["at_bat_number"] == 1 and row["pitch_number"] == 4:
                    row["pitch_type"] = "KC"
            write_csv(root, "data/raw/savant_data.csv", rows)
            # Drop only outcome labels. The same test games now have different row counts.
            for row in rows:
                if row["game_pk"] == 107 and row["at_bat_number"] <= 10 and row["pitch_number"] == 2:
                    row["delta_run_exp"] = None
            write_csv(root, "data/raw/savant_data_with_outcome.csv", rows)
            run_step(root, "game_split")
            manifest = root / "data/processed/game_split.csv"
            manifest_frame = pd.read_csv(manifest)
            expected_test = set(manifest_frame.loc[manifest_frame["split"] == "test", "game_pk"])
            self.assertEqual(expected_test, {107, 108})

            def run(step):
                return run_step(root, step, game_split=manifest)

            for step in ("preprocess", "make_sequence", "make_outcome_sequence"):
                run(step)
            for model in ("lstm", "transformer"):
                log = run(f"train_{model}")
                self.assertIn("Validation: excluded 1 of 300 rows", log)
                self.assertIn("Test: excluded 1 of 600 rows", log)
            for step in ("outcome_empirical_recommender", "export_serving_artifacts"):
                run(step)
            self.assertIn("Test: (600,", run("baseline"))

            for model in ("lstm", "transformer"):
                predictions = pd.read_csv(root / f"results/{model}_weighted_macro_predictions.csv")
                self.assertEqual(len(predictions), 599)
                self.assertEqual(set(predictions["game_pk"]), expected_test)
                self.assertNotIn("KC", predictions["true_pitch_type"].tolist())
            recommendations = pd.read_csv(root / "results/outcome_empirical_recommendations.csv")
            self.assertEqual(len(recommendations), 590)
            self.assertEqual(set(recommendations["game_pk"]), expected_test)
            unmatched = recommendations.loc[recommendations["lstm_top3_pitch_type"].isna()]
            self.assertEqual(len(unmatched), 1)
            self.assertEqual(unmatched["game_pk"].tolist(), [108])

            outcomes = pd.read_csv(root / "data/processed/outcome_sequence_data.csv")
            fit = outcomes.loc[~outcomes["game_pk"].isin(expected_test)]
            metrics = pd.read_csv(root / "results/outcome_empirical_eval.csv").iloc[0]
            with (root / "models/outcome_ranker_artifact.pkl").open("rb") as stream:
                artifact = pickle.load(stream)
            self.assertAlmostEqual(metrics["global_mean"], fit["delta_run_exp"].mean())
            self.assertAlmostEqual(artifact["global_mean"], metrics["global_mean"])
