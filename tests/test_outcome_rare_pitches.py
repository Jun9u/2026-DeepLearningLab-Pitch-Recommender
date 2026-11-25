from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from tests.support import pitch, run_cli_in_process, write_csv


class OutcomeRarePitchTests(unittest.TestCase):
    def test_game_split_rarity_defaults_to_ranker_fit_games(self):
        game_pitches = [
            ["SL"] * 499 + ["FF"] * 494 + ["KN"] * 500,
            ["FF"] * 6,
            ["SL", "FF", "FF", "SL"] + ["CH"] * 500,
        ]
        rows = [
            pitch(index // 2 + 1, index % 2 + 1, code, game_pk=100 + day, game_date=f"2025-06-0{day}")
            for day, pitches in enumerate(game_pitches, start=1)
            for index, code in enumerate(pitches)
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "games.csv"
            manifest.write_text(
                "game_pk,game_date,split\n101,2025-06-01,train\n102,2025-06-02,validation\n103,2025-06-03,test\n",
            )
            write_csv(root, "data/raw/savant_data_with_outcome.csv", rows)
            result_path = root / "data/processed/outcome_sequence_data.csv"
            run_cli_in_process(root, "make_outcome_sequence")
            legacy = pd.read_csv(result_path)
            run_cli_in_process(root, "make_outcome_sequence", game_split=manifest)
            default = pd.read_csv(result_path)
            run_cli_in_process(root, "make_outcome_sequence", "--train-only-rare-pitches", game_split=manifest)
            fit_only = pd.read_csv(result_path)

        keys = ["game_pk", "at_bat_number", "pitch_number", "action_pitch_type_original", "delta_run_exp"]
        pd.testing.assert_frame_equal(default, fit_only)
        pd.testing.assert_frame_equal(legacy[keys], fit_only[keys])
        self.assertEqual(set(legacy["action_pitch_type"]), {"FF", "SL", "CH", "OTHER"})
        self.assertEqual(set(fit_only["action_pitch_type"]), {"FF", "OTHER"})
        before = legacy.loc[legacy["game_pk"] == 103].iloc[0]
        after = fit_only.loc[fit_only["game_pk"] == 103].iloc[0]
        self.assertEqual((before["prev_pitch_type"], after["prev_pitch_type"]), ("SL", "OTHER"))
        self.assertEqual(after["action_pitch_type"], "FF")

    def test_train_only_mode_requires_a_game_manifest(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires PITCH_GAME_SPLIT"):
                run_cli_in_process(Path(directory), "make_outcome_sequence", "--train-only-rare-pitches")
