import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException, Request

from web.backend import app as api
from web.backend.schemas import CompareRequest, RecommendRequest


class AppReadinessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.lstm = root / "lstm_serving_artifact.pt"
        self.outcome = root / "outcome_ranker_artifact.pkl"
        self.enterContext(patch.multiple(api, LSTM_ARTIFACT_PATH=self.lstm, OUTCOME_ARTIFACT_PATH=self.outcome))
        self.loader = Mock()
        service_module = SimpleNamespace(PitchRecommendationService=self.loader)
        self.enterContext(patch.dict(sys.modules, {"web.backend.service": service_module}))

    async def test_missing_file_keeps_service_unavailable_for_this_startup(self):
        self.lstm.touch()

        async with api.lifespan(api.app) as state:
            request = Request({"type": "http", "state": state})
            response = api.health(request)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(json.loads(response.body), {
                "status": "not_ready", "missing_artifacts": [self.outcome.name],
            })
            with self.assertRaises(HTTPException) as captured:
                api.require_service(request)
            self.assertEqual(captured.exception.status_code, 503)
            self.outcome.touch()
            self.assertEqual(api.health(request).status_code, 503)

        self.loader.assert_not_called()
        self.assertEqual(api.live(), {"status": "ok"})

    async def test_loads_both_files_once_and_routes_use_the_service(self):
        self.lstm.touch()
        self.outcome.touch()
        service = self.loader.return_value
        service.device = "cpu"
        service.pitcher_vocab = {123: 1}
        service.pitcher_top_pitches = {123: ["FF"]}
        service.valid_zones = ["Z5"]
        service.supported_pitch_types = service.global_top_pitches = ["FF"]
        service.stand_vocab = service.throws_vocab = {"R": 1}

        async with api.lifespan(api.app) as state:
            request = Request({"type": "http", "state": state})
            self.assertEqual(api.health(request), {
                "status": "ok", "device": "cpu", "num_pitchers_lstm_vocab": 1, "num_pitchers_outcome": 1,
                "valid_zones": ["Z5"], "supported_pitch_types": ["FF"],
            })
            self.assertEqual(api.meta(request)["global_top_pitches"], ["FF"])
            payload = {"pitcher": 123, "balls": 1, "strikes": 2, "outs_when_up": 1, "inning": 1}
            recommendation = RecommendRequest(**payload)
            comparison = CompareRequest(**payload, action_pitch_type="FF", action_zone="Z5")
            self.assertIs(api.recommend(recommendation, request), service.recommend.return_value)
            self.assertIs(api.compare(comparison, request), service.compare.return_value)

        self.loader.assert_called_once_with(self.lstm, self.outcome)
        service.recommend.assert_called_once_with(recommendation)
        service.compare.assert_called_once_with(comparison)

    async def test_loader_errors_fail_startup(self):
        self.lstm.touch()
        self.outcome.touch()
        self.loader.side_effect = ValueError("invalid artifact")

        with self.assertRaisesRegex(ValueError, "invalid artifact"):
            async with api.lifespan(api.app):
                self.fail("Startup succeeded with a corrupt artifact")
