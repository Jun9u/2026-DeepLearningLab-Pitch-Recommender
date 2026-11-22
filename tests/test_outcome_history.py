from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from tests.support import REPO, command, pitch, write_csv


class OutcomeHistoryTests(unittest.TestCase):
    script = str(REPO / "src/make_outcome_sequence.py")

    def test_known_unlabeled_pitches_remain_in_history_only(self):
        rows = [pitch(1, number, "FF", zone=number) for number in range(1, 9)]
        rows[2]["delta_run_exp"] = None
        rows[4]["zone"] = None
        rows[5]["pitch_type"] = None
        rows.extend([
            pitch(2, 1, "CH", zone=4, delta_run_exp=None),
            pitch(2, 2, "FF", zone=2),
            pitch(2, 3, "FF", zone=3),
        ])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "data/raw/savant_data_with_outcome.csv", reversed(rows))
            data = root / "data/processed/outcome_sequence_data.csv"
            command(root, self.script)
            before = pd.read_csv(data).set_index(["at_bat_number", "pitch_number"])
            command(root, self.script, "--keep-unlabeled-history")
            after = pd.read_csv(data).set_index(["at_bat_number", "pitch_number"])

        self.assertEqual(before.index.tolist(), [(1, 2), (1, 4), (1, 7), (1, 8), (2, 3)])
        self.assertEqual(after.index.tolist(), [(1, 2), (1, 4), (1, 7), (1, 8), (2, 2), (2, 3)])
        self.assertEqual(after.loc[(1, 4), "seq_zone"], "ZPAD|ZPAD|Z1|Z2|Z3")
        self.assertEqual(after.loc[(1, 8), "seq_zone"], "Z2|Z3|Z4|ZUNK|Z7")
        self.assertEqual(after.loc[(1, 7), "prev_zone"], "ZUNK")
        self.assertEqual(after.loc[(2, 2), "seq_zone"], "ZPAD|ZPAD|ZPAD|ZPAD|Z4")
        self.assertEqual(after.loc[(2, 2), "seq_len"], 1)
        self.assertTrue(after["delta_run_exp"].notna().all())
        self.assertNotIn("ZUNK", set(after["action_zone"]))

    def test_unlabeled_history_does_not_count_toward_rarity_cutoff(self):
        fit = [("SL", 0.1)] * 499 + [("SL", None)] + [("FF", 0.2)] * 500
        games = [fit, [("FF", 0.3)] * 2, [("SL", 0.4), ("FF", 0.5), ("FF", 0.5), ("SL", 0.4)]]
        rows = [
            pitch(
                index // 2 + 1, index % 2 + 1, code,
                game_pk=100 + day, game_date=f"2025-06-0{day}", delta_run_exp=outcome,
            )
            for day, game in enumerate(games, start=1)
            for index, (code, outcome) in enumerate(game)
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "games.csv"
            manifest.write_text(
                "game_pk,game_date,split\n101,2025-06-01,train\n102,2025-06-02,validation\n103,2025-06-03,test\n",
            )
            write_csv(root, "data/raw/savant_data_with_outcome.csv", rows)
            command(root, self.script, "--train-only-rare-pitches", "--keep-unlabeled-history", game_split=manifest)
            data = pd.read_csv(root / "data/processed/outcome_sequence_data.csv")

        test = data.loc[data["game_pk"] == 103]
        self.assertEqual(test["action_pitch_type"].tolist(), ["FF", "OTHER"])
        self.assertEqual(test["prev_pitch_type"].tolist(), ["OTHER", "FF"])
