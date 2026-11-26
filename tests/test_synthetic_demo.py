from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.synthetic_pitches import DEMO_MARKER, EXAMPLE_PITCHER
from tests.support import run_cli_in_process


class SyntheticDemoTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.run_dir = self.root / "demo"
        self.run_step = self.enterContext(patch("subprocess.run"))

    def launch(self):
        run_cli_in_process(self.root, "synthetic_demo", "--run-dir", str(self.run_dir),
                           game_split=self.root / "unrelated.csv")

    def test_generates_fictional_data_and_runs_pipeline_in_the_demo_directory(self):
        self.launch()

        raw_dir = self.run_dir / "data/raw"
        rows = pd.read_csv(raw_dir / "savant_data.csv")
        self.assertEqual(len(rows), 3600)
        self.assertIn(EXAMPLE_PITCHER, set(rows["pitcher"]))
        self.assertEqual(set(rows["zone"]), set(range(1, 10)) | {11, 12, 13, 14})
        self.assertEqual((raw_dir / "savant_data.csv").read_bytes(),
                         (raw_dir / "savant_data_with_outcome.csv").read_bytes())
        self.assertTrue((self.run_dir / DEMO_MARKER).is_file())

        steps = self.run_step.call_args_list
        self.assertEqual([step.args[0][-1] for step in steps], [
            "src.game_split", "src.preprocess", "src.make_sequence", "src.train_lstm",
            "src.make_outcome_sequence", "src.export_serving_artifacts",
        ])
        for step in steps:
            self.assertEqual(step.kwargs["env"]["PITCH_RUN_DIR"], str(self.run_dir))
            self.assertEqual(step.kwargs["env"]["PITCH_GAME_SPLIT"],
                             str(self.run_dir / "data/processed/game_split.csv"))
            self.assertTrue(step.kwargs["check"])

    def test_existing_directory_is_preserved(self):
        self.run_dir.mkdir()
        existing = self.run_dir / "keep.txt"
        existing.write_text("keep", encoding="utf-8")

        with self.assertRaises(SystemExit) as captured:
            self.launch()

        self.assertEqual(captured.exception.code, 2)
        self.assertEqual(list(self.run_dir.iterdir()), [existing])
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.run_step.assert_not_called()

    def test_pipeline_failure_leaves_no_ready_marker(self):
        self.run_step.side_effect = subprocess.CalledProcessError(1, "step")

        with self.assertRaises(subprocess.CalledProcessError):
            self.launch()

        self.assertFalse((self.run_dir / DEMO_MARKER).exists())
