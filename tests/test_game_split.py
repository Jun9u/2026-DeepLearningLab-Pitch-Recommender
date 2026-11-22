import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.game_split import build_game_split, split_indices


class GameSplitTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame({
            "game_date": [f"2025-06-{day:02d}" for day in range(1, 6) for _ in range(2)],
            "game_pk": list(range(101, 111)),
        })

    def test_counts_games_once_and_keeps_each_date_together(self):
        repeated_games = pd.concat([self.games.iloc[[-1]]]*20 + [self.games.iloc[::-1]], ignore_index=True)
        split = build_game_split(repeated_games)

        self.assertEqual(split["game_pk"].tolist(), list(range(101, 111)))
        self.assertEqual(split["split"].tolist(), ["train"] * 6 + ["validation"] * 2 + ["test"] * 2)

    def test_all_consumers_share_test_games_even_when_rows_are_missing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            build_game_split(self.games).to_csv(path, index=False)
            full = pd.concat([self.games, self.games], ignore_index=True)
            filtered = full.drop(index=[8, 9, 10])
            with patch.dict(os.environ, PITCH_GAME_SPLIT=str(path)):
                train, valid, test = split_indices(full)
                fit, empty, held_out = split_indices(filtered, with_validation=False)

        self.assertEqual(set(full.iloc[train]["game_pk"]), set(range(101, 107)))
        self.assertEqual(set(full.iloc[valid]["game_pk"]), {107, 108})
        self.assertEqual(set(filtered.iloc[fit]["game_pk"]), set(range(101, 109)))
        self.assertTrue(filtered.iloc[empty].empty)
        self.assertEqual(set(full.iloc[test]["game_pk"]), {109, 110})
        self.assertEqual(set(filtered.iloc[held_out]["game_pk"]), {109, 110})

    def test_rejects_unknown_games_and_conflicting_dates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            build_game_split(self.games).to_csv(path, index=False)
            with patch.dict(os.environ, PITCH_GAME_SPLIT=str(path)):
                for field, value in (("game_pk", 999), ("game_date", "2025-06-17")):
                    with self.subTest(field=field):
                        changed = self.games.copy()
                        changed.loc[0, field] = value
                        with self.assertRaisesRegex(ValueError, "match the game split manifest"):
                            split_indices(changed)

    def test_requires_three_dates(self):
        with self.assertRaisesRegex(ValueError, "at least three dates"):
            build_game_split(self.games.iloc[:4])

    def test_default_keeps_existing_row_boundaries(self):
        with patch.dict(os.environ, PITCH_GAME_SPLIT=""):
            self.assertEqual(split_indices(self.games), (slice(0, 7), slice(7, 8), slice(8, None)))
            self.assertEqual(split_indices(self.games, with_validation=False),
                             (slice(0, 8), slice(8, 8), slice(8, None)))
