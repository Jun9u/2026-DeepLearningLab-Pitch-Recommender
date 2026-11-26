import importlib.util
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

from src.empirical_ranker import COUNT_ZONE_MODE
from tests.support import REPO


def test_serving_uses_selected_scorer_with_an_existing_artifact(tmp_path):
    weights = {
        **dict.fromkeys(("model_state_dict", "pitch_vocab", "zone_vocab", "pitcher_vocab", "stand_vocab",
                         "throws_vocab", "numeric_cols", "target_to_idx"), {}),
        "idx_to_target": {0: "FF"}, "scaler_mean": [0.0], "scaler_scale": [1.0],
    }
    # Loading the classifier is covered by the integration suite. Supply its metadata here.
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), device=str,
                            load=lambda *args, **kwargs: weights)
    trainer = SimpleNamespace(load_serving_model=lambda *args: None)
    spec = importlib.util.spec_from_file_location("web.backend._service_under_test", REPO / "web/backend/service.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "src.train_lstm": trainer}):
        spec.loader.exec_module(module)

    lstm = tmp_path / "lstm.pt"
    lstm.touch()
    outcome = tmp_path / "outcome.pkl"
    artifact = {
        "global_mean": 0.3, "alpha": 4.0, "thresholds": {}, "valid_zones": ["Z5"],
        "pitcher_top_pitches": {123: ["FF"]}, "global_top_pitches": ["FF"],
        "stats": {
            "count_action_zone": {(3, 2, "FF", "Z5"): {"sum": 0.5, "count": 1}},
            "action_zone": {("FF", "Z5"): {"mean": 0.1, "count": 40}},
            "action": {("FF",): {}},
        },
    }
    outcome.write_bytes(pickle.dumps(artifact))

    service = module.PitchRecommendationService(lstm, outcome, ranker_mode=COUNT_ZONE_MODE)

    request = SimpleNamespace(pitcher=123, balls=3, strikes=2)
    assert service._predict_delta(request, "ff", "z5") == (3 / 26, "count+action+zone", 1)
