import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pandas as pd

from tests.support import (
    REPO,
    api_server,
    command,
    pipeline_env,
    pipeline_pitches,
    pitch,
    run_step,
    sparse_outcome_rows,
    stage_web,
    uvicorn_args,
    write_csv,
)

PUBLIC_RESULTS = {
    "baseline_results.csv": "62468d69c347170fa4d8efbbbfd83d7ece1ccffc7650e5cf1a40c64096a36449",
    "lstm_weighted_macro_result.csv": "f29509374171fd672860d27bec4a561ba4abffc51431ba04e58357264055dc82",
    "outcome_empirical_eval.csv": "3d5b0267800bc56034d2f657b0e56a1220e8accd2bfb708d18e40a684215f314",
    "outcome_recommendation_sample.csv": "c6931fd84cb23c2b17f4c5db746245b7df41a69260dc85a96dac4abf3ac8759a",
    "transformer_weighted_macro_result.csv": "f1a7c056aa3f34bb4cb18d3c401632a70b62d14523f4b636488e1a18cd55c10a",
}
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
            write_csv(root, "fixture.csv", pipeline_pitches())
            # Replace only the external data request; the collection script still writes both schemas.
            command(root, "-c", """
import runpy
import sys
from unittest.mock import patch

import pandas as pd

with patch("pybaseball.statcast", return_value=pd.read_csv("fixture.csv")):
    runpy.run_path(sys.argv[1], run_name="__main__")
""", str(REPO / "src/download_statcast.py"))
            for step in (
                "check_csv", "check_data_summary", "preprocess", "make_sequence", "baseline",
                "train_lstm", "train_transformer", "make_outcome_sequence",
                "outcome_empirical_recommender", "export_serving_artifacts",
            ):
                run_step(root, step)

            for filename in ("sequence_data.csv", "outcome_sequence_data.csv"):
                self.assertEqual(len(pd.read_csv(root / "data/processed" / filename)), 2700)
            self.assertEqual(len(pd.read_csv(root / "outputs/baseline_results.csv")), 6)
            for model in ("lstm", "transformer"):
                predictions = pd.read_csv(root / f"outputs/{model}_weighted_macro_predictions.csv")
                self.assertEqual(len(predictions), 540)
                self.assertTrue(predictions["top3_pitch_type"].str.count(r"\|").eq(2).all())
                metrics = pd.read_csv(root / f"outputs/{model}_weighted_macro_result.csv").iloc[0]
                self.assertAlmostEqual(metrics["accuracy"],
                                       predictions["true_pitch_type"].eq(predictions["pred_pitch_type"]).mean())
            ranking = pd.read_csv(root / "outputs/outcome_empirical_eval.csv").iloc[0]
            self.assertEqual(ranking["test_rows"], 540)
            stage_web(root)
            with api_server(root) as base:
                self.check_model_api(base)

            # Keep the trained LSTM and rebuild the ranker with sparse groups.
            write_csv(root, "data/processed/outcome_sequence_data.csv", sparse_outcome_rows())
            run_step(root, "export_serving_artifacts")
            with api_server(root) as base:
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
