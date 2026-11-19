import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pandas as pd

from tests.support import (
    PUBLIC_RESULTS,
    REPO,
    api_server,
    command,
    pipeline_env,
    pipeline_pitches,
    run_step,
    sparse_outcome_rows,
    stage_web,
    uvicorn_args,
    write_csv,
)


REQUEST = {
    "pitcher": 900001,
    "balls": 1,
    "strikes": 1,
    "outs_when_up": 1,
    "inning": 5,
    "prev_pitch_types": ["FF"],
    "prev_zones": ["Z5"],
}
HTTP = build_opener(ProxyHandler({}))


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    query = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    try:
        response = HTTP.open(query, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        if response.headers.get_content_type() == "application/json":
            return response.status, json.load(response)
        return response.status, response.read()


class BaselineIntegrationTests(unittest.TestCase):
    def test_missing_artifacts_prevent_server_start(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stage_web(root)
            result = subprocess.run(
                uvicorn_args(), cwd=root, env=pipeline_env(root), timeout=45,
                capture_output=True, encoding="utf-8",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileNotFoundError", result.stderr)
        self.assertIn("lstm_serving_artifact.pt", result.stderr)

    def test_synthetic_workflow_and_model_api(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            launch_dir = Path("runs/synthetic")
            run_root = root / launch_dir
            published = run_root / "outputs"
            published.mkdir(parents=True)
            for name in PUBLIC_RESULTS:
                shutil.copyfile(REPO / "outputs" / name, published / name)
            published_contents = {file.name: file.read_bytes() for file in published.glob("*.csv")}
            write_csv(root, "fixture.csv", pipeline_pitches())
            # Replace only the external data request; the collection script still writes both schemas.
            command(root, "-c", """
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pandas as pd

# Match the import path used when Python runs a script directly.
sys.path.insert(0, str(Path(sys.argv[1]).parent))
with patch("pybaseball.statcast", return_value=pd.read_csv("fixture.csv")):
    runpy.run_path(sys.argv[1], run_name="__main__")
""", str(REPO / "src/download_statcast.py"), run_directory=launch_dir)
            for step in (
                "check_csv", "check_data_summary", "preprocess", "make_sequence", "baseline",
                "train_lstm", "train_transformer", "make_outcome_sequence",
                "outcome_empirical_recommender", "export_serving_artifacts",
            ):
                run_step(root, step, run_directory=launch_dir)

            for filename in ("sequence_data.csv", "outcome_sequence_data.csv"):
                self.assertEqual(len(pd.read_csv(run_root / "data/processed" / filename)), 2700)
            self.assertEqual(len(pd.read_csv(run_root / "results/baseline_results.csv")), 6)
            for model in ("lstm", "transformer"):
                predictions = pd.read_csv(run_root / f"results/{model}_weighted_macro_predictions.csv")
                self.assertEqual(len(predictions), 540)
                self.assertTrue(predictions["top3_pitch_type"].str.count(r"\|").eq(2).all())
                metrics = pd.read_csv(run_root / f"results/{model}_weighted_macro_result.csv").iloc[0]
                self.assertAlmostEqual(metrics["accuracy"],
                                       predictions["true_pitch_type"].eq(predictions["pred_pitch_type"]).mean())
            ranking = pd.read_csv(run_root / "results/outcome_empirical_eval.csv").iloc[0]
            self.assertEqual(ranking["test_rows"], 540)
            recommendations = pd.read_csv(run_root / "results/outcome_empirical_recommendations.csv")
            self.assertTrue(recommendations["lstm_top3_pitch_type"].notna().all())

            for directory_name in ("data", "results", "models"):
                self.assertFalse((root / directory_name).exists(), directory_name)
            stage_web(root)
            with api_server(root, run_directory=launch_dir) as base:
                self.check_model_api(base)

            # Keep the trained LSTM and rebuild the ranker with sparse groups.
            write_csv(run_root, "data/processed/outcome_sequence_data.csv", sparse_outcome_rows())
            run_step(root, "export_serving_artifacts", run_directory=launch_dir)
            with api_server(root, run_directory=launch_dir) as base:
                status, response = request(
                    base, "/recommend", {**REQUEST, "candidate_pitch_types": ["SL", "FF"]},
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    [(item["pitch_type"], item["zone"], item["source"], item["source_count"])
                     for item in response["recommendations"]],
                    [("SL", "Z1", "global", 0), ("SL", "Z2", "global", 0),
                     ("FF", "Z1", "global", 0), ("FF", "Z2", "global", 0)],
                )

            self.assertEqual(
                {file.name: file.read_bytes() for file in published.glob("*.csv")}, published_contents,
            )

    def check_model_api(self, base):
        self.assertEqual(request(base, "/")[0], 200)
        status, health = request(base, "/health")
        self.assertEqual((status, health["status"]), (200, "ok"))
        self.assertEqual(health["num_pitchers_outcome"], 4)
        _, meta = request(base, "/meta")
        self.assertEqual(set(meta["supported_pitch_types"]), {"FF", "SL", "CH"})
        self.assertEqual(meta["valid_zones"], [f"Z{zone}" for zone in range(1, 10)])
        _, schema = request(base, "/openapi.json")
        self.assertEqual(set(schema["paths"]["/recommend"]), {"post"})

        status, recommendation = request(base, "/recommend", REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(len(recommendation["lstm_top3"]), 3)
        ranked = recommendation["recommendations"]
        self.assertEqual(len(ranked), 5)
        scores = [item["pred_delta_run_exp"] for item in ranked]
        self.assertEqual(scores, sorted(scores))

        best = ranked[0]
        status, comparison = request(base, "/compare", {
            **REQUEST, "action_pitch_type": best["pitch_type"], "action_zone": best["zone"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(comparison["best_recommendation"], best)
        self.assertEqual(comparison["comparison"], {
            "selected_minus_best": 0, "is_selected_better_or_equal": True,
        })

        higher = next((item for item in ranked if item["pred_delta_run_exp"] > scores[0]), None)
        self.assertIsNotNone(higher, "The fixture must contain a recommendation with a higher score.")
        status, comparison = request(base, "/compare", {
            **REQUEST, "action_pitch_type": higher["pitch_type"], "action_zone": higher["zone"],
        })
        self.assertEqual(status, 200)
        difference = comparison["comparison"]["selected_minus_best"]
        self.assertGreater(difference, 0)
        self.assertAlmostEqual(difference, higher["pred_delta_run_exp"] - scores[0])
        self.assertFalse(comparison["comparison"]["is_selected_better_or_equal"])

        _, manual = request(base, "/recommend", {**REQUEST, "candidate_pitch_types": [" ch ", "X", "CH"]})
        self.assertEqual(manual["candidate_pitches"], ["CH"])
        _, fallback = request(base, "/recommend", {**REQUEST, "candidate_pitch_types": ["X"], "prev_zones": []})
        self.assertTrue(fallback["recommendations"])
        self.assertIn("prev_pitch_types", fallback["warnings"][0])
        status, invalid = request(base, "/compare", {**REQUEST, "action_pitch_type": "X", "action_zone": "Z5"})
        self.assertEqual(status, 200)
        self.assertIn("error", invalid)
        status, invalid = request(base, "/recommend", {**REQUEST, "balls": 4})
        self.assertEqual(status, 422)
        self.assertEqual(invalid["detail"][0]["loc"], ["body", "balls"])
