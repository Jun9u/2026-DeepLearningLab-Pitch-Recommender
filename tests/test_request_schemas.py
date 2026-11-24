import pytest
from pydantic import ValidationError

from web.backend.schemas import CompareRequest, RecommendRequest


BASE_REQUEST = {"pitcher": 123, "balls": 1, "strikes": 2, "outs_when_up": 0, "inning": 1}


@pytest.fixture(params=[RecommendRequest, CompareRequest])
def validate_request(request):
    model = request.param
    payload = BASE_REQUEST.copy()
    if model is CompareRequest:
        payload.update(action_pitch_type="FF", action_zone="Z5")

    def validate(history):
        return model.model_validate(payload | history)

    return validate


@pytest.mark.parametrize("history", [
    {},
    {"prev_pitch_types": [], "prev_zones": []},
    {"prev_pitch_types": ["FF", "OTHER"], "prev_zones": ["Z1", "Z5"]},
    {"prev_pitch_types": ["FF"] * 6, "prev_zones": ["Z5"] * 6},
])
def test_accepts_empty_or_paired_history(validate_request, history):
    parsed = validate_request(history)
    assert parsed.prev_pitch_types == history.get("prev_pitch_types", [])
    assert parsed.prev_zones == history.get("prev_zones", [])


@pytest.mark.parametrize("history", [
    {"prev_pitch_types": ["FF", "SL"], "prev_zones": ["Z5"]},
    {"prev_pitch_types": ["FF"], "prev_zones": ["Z5", "Z4"]},
    {"prev_pitch_types": ["FF"]},
    {"prev_zones": ["Z5"]},
])
def test_rejects_unpaired_history(validate_request, history):
    with pytest.raises(ValidationError) as exc:
        validate_request(history)

    assert [error["type"] for error in exc.value.errors()] == ["pitch_history_length_mismatch"]
