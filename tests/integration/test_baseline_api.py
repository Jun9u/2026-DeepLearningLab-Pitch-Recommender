from io import BytesIO
import json
import pickle
from pathlib import Path
import shutil
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
    pipeline_pitches,
    run_step,
    sparse_outcome_rows,
    stage_web,
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
    def test_ranker_loader_accepts_containers_and_rejects_other_objects(self):
        from web.backend.service import RankerArtifactUnpickler

        expected = {"stats": {(1, "FF", "Z1"): {"count": 1, "mean": 0.2}}}
        self.assertEqual(RankerArtifactUnpickler(BytesIO(pickle.dumps(expected))).load(), expected)
        with self.assertRaises(pickle.UnpicklingError):
            RankerArtifactUnpickler(BytesIO(pickle.dumps(Path("model")))).load()

    def test_missing_artifacts_leave_static_routes_available(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stage_web(root)
            with api_server(root) as base:
                preflight = Request(base + "/recommend", method="OPTIONS", headers={
                    "Origin": "https://example.com", "Access-Control-Request-Method": "POST",
                })
                with self.assertRaises(HTTPError) as response:
                    HTTP.open(preflight, timeout=10)
                with response.exception as error:
                    self.assertNotIn("Access-Control-Allow-Origin", error.headers)

                for path in ("/", "/assets/styles.css", "/assets/app.js", "/docs"):
                    with self.subTest(path=path):
                        self.assertEqual(request(base, path)[0], 200)
                self.assertEqual(request(base, "/live"), (200, {"status": "ok"}))
                self.assertEqual(request(base, "/health"), (503, {
                    "status": "not_ready",
                    "missing_artifacts": ["lstm_serving_artifact.pt", "outcome_ranker_artifact.pkl"],
                }))
                for path, payload in (
                    ("/meta", None),
                    ("/recommend", REQUEST),
                    ("/compare", {**REQUEST, "action_pitch_type": "FF", "action_zone": "Z5"}),
                ):
                    with self.subTest(path=path):
                        status, response = request(base, path, payload)
                        self.assertEqual(status, 503)
                        self.assertIn("서버를 재시작", response["detail"])

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
script = sys.argv.pop(1)
sys.path.insert(0, str(Path(script).parent))
with patch("pybaseball.statcast", return_value=pd.read_csv("fixture.csv")) as statcast:
    runpy.run_path(script, run_name="__main__")
    statcast.assert_called_once_with("2025-06-01", "2025-06-09")
""", str(REPO / "src/download_statcast.py"), "--start-date", "2025-06-01", "--end-date", "2025-06-09",
                    run_directory=launch_dir)
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

            for directory_name in ("data", "results", "models", "outputs"):
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
        status, html = request(base, "/")
        self.assertEqual(status, 200)
        for name, content_types in (
            ("styles.css", {"text/css"}),
            ("app.js", {"text/javascript", "application/javascript"}),
        ):
            with self.subTest(asset=name):
                self.assertIn(f'"assets/{name}"'.encode(), html)
                with HTTP.open(f"{base}/assets/{name}", timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(response.headers.get_content_type(), content_types)
                    self.assertEqual(response.read(), (REPO / "web/frontend/assets" / name).read_bytes())
        self.assertEqual(request(base, "/assets/missing.js")[0], 404)
        self.assertEqual(request(base, "/live"), (200, {"status": "ok"}))
        status, health = request(base, "/health")
        self.assertEqual((status, health["status"]), (200, "ok"))
        self.assertEqual(health["num_pitchers_outcome"], 4)
        _, meta = request(base, "/meta")
        self.assertEqual(set(meta["supported_pitch_types"]), {"FF", "SL", "CH"})
        self.assertEqual(meta["valid_zones"], [f"Z{zone}" for zone in range(1, 10)])
        _, schema = request(base, "/openapi.json")
        self.assertEqual(set(schema["paths"]["/recommend"]), {"post"})
        compare_responses = schema["paths"]["/compare"]["post"]["responses"]
        validation_schema, action_error_schema = compare_responses["422"]["content"]["application/json"]["schema"]["anyOf"]
        self.assertEqual(validation_schema["$ref"], "#/components/schemas/HTTPValidationError")
        self.assertIn("HTTPValidationError", schema["components"]["schemas"])
        self.assertEqual(action_error_schema["required"], ["error"])

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
        self.assertFalse(manual["warnings"])
        status, fallback = request(base, "/recommend", {
            **REQUEST, "candidate_pitch_types": ["X"], "prev_pitch_types": [], "prev_zones": [],
        })
        self.assertEqual(status, 200)
        self.assertTrue(fallback["recommendations"])
        self.assertIn("이전 투구 정보가 없습니다", fallback["warnings"][0])
        warning = "수동 후보에 지원하는 구종이 없어 자동으로 후보를 선택했습니다."
        self.assertEqual(fallback["warnings"][1:], [warning])
        status, fallback_comparison = request(base, "/compare", {
            **REQUEST, "candidate_pitch_types": ["X"], "action_pitch_type": "FF", "action_zone": "Z5",
        })
        self.assertEqual(status, 200)
        self.assertIsNotNone(fallback_comparison["best_recommendation"])
        self.assertEqual(fallback_comparison["warnings"], [warning])
        mismatched = {**REQUEST, "prev_zones": []}
        for path, payload in (
            ("/recommend", mismatched),
            ("/compare", {**mismatched, "action_pitch_type": "FF", "action_zone": "Z5"}),
        ):
            with self.subTest(path=path):
                status, error = request(base, path, payload)
                self.assertEqual(status, 422)
                detail = error["detail"][0]
                self.assertEqual(detail["type"], "pitch_history_length_mismatch")
                self.assertEqual(detail["loc"], ["body"])
                self.assertIn("같은 개수", detail["msg"])
        invalid_actions = (
            (" other ", "Z5", {"error": "유효하지 않은 구종입니다.", "input_pitch_type": " other "}),
            (" x ", "Z5", {
                "error": "지원하지 않는 구종입니다.",
                "input_pitch_type": "X",
                "supported_pitch_types": meta["supported_pitch_types"],
            }),
            ("FF", "99", {
                "error": "지원하지 않는 zone입니다.", "input_zone": "Z99", "valid_zones": meta["valid_zones"],
            }),
            ("FF", "Infinity", {
                "error": "지원하지 않는 zone입니다.", "input_zone": "ZUNK", "valid_zones": meta["valid_zones"],
            }),
            ("FF", "1.9", {
                "error": "지원하지 않는 zone입니다.", "input_zone": "ZUNK", "valid_zones": meta["valid_zones"],
            }),
        )
        for pitch, zone, error in invalid_actions:
            with self.subTest(pitch=pitch, zone=zone):
                self.assertEqual(request(base, "/compare", {
                    **REQUEST, "action_pitch_type": pitch, "action_zone": zone,
                }), (422, error))

        status, nonfinite_history = request(base, "/recommend", {**REQUEST, "prev_zones": ["Infinity"]})
        self.assertEqual(status, 200)
        status, unknown_history = request(base, "/recommend", {**REQUEST, "prev_zones": ["ZUNK"]})
        self.assertEqual(status, 200)
        self.assertEqual(nonfinite_history["lstm_top3"], unknown_history["lstm_top3"])
        self.assertEqual(nonfinite_history["recommendations"], unknown_history["recommendations"])

        status, missing_action = request(base, "/compare", {**REQUEST, "action_pitch_type": "FF"})
        self.assertEqual(status, 422)
        self.assertEqual(missing_action["detail"][0]["loc"], ["body", "action_zone"])
        status, invalid = request(base, "/recommend", {**REQUEST, "balls": 4})
        self.assertEqual(status, 422)
        self.assertEqual(invalid["detail"][0]["loc"], ["body", "balls"])
