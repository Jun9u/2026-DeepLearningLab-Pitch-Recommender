"""Small tests that run without downloaded Statcast data or model artifacts."""

import math

import pandas as pd

from src.make_sequence import left_pad, map_rare_token_sequence, zone_to_token
from src.outcome_empirical_recommender import (
    parse_top3,
    smoothed_mean,
    zone_sort_key,
)
from src.sequence_helpers import add_runner_flags, iter_pitch_histories


def test_left_pad_short_sequence():
    assert left_pad(["FF", "SL"], 4, "PAD") == ["PAD", "PAD", "FF", "SL"]


def test_left_pad_truncates_oldest_values():
    assert left_pad([1, 2, 3, 4], 2, 0) == [3, 4]


def test_rare_pitch_mapping():
    assert map_rare_token_sequence("FF|CS|SL", {"CS"}) == "FF|OTHER|SL"


def test_zone_token_normalization():
    assert zone_to_token(3.0) == "Z3"
    assert zone_to_token(float("nan")) == "ZUNK"
    assert zone_to_token("unknown") == "Zunknown"


def test_zone_sort_key():
    assert zone_sort_key("Z12") == 12
    assert zone_sort_key("ZUNK") == 999


def test_smoothing_uses_global_mean():
    assert math.isclose(smoothed_mean(10.0, 2, 5.0, alpha=2.0), 5.0)


def test_top3_parsing():
    assert parse_top3("FF|SL|nan|CH") == ["FF", "SL", "CH"]
    assert parse_top3(float("nan")) == []


def test_runner_flags_track_each_base_without_overwriting_runner_ids():
    source = pd.DataFrame({"on_1b": [11, None], "on_2b": [None, 22], "on_3b": [33, None]})
    frame = source.copy()

    add_runner_flags(frame)

    pd.testing.assert_frame_equal(frame[source.columns], source)
    assert frame[["runner_1b", "runner_2b", "runner_3b"]].values.tolist() == [[1, 0, 1], [0, 1, 0]]


def test_pitch_history_respects_game_and_at_bat_boundaries():
    frame = pd.DataFrame(
        [
            ("2025-06-02", 20, 1, 2, "CH", "Z9"),
            ("2025-06-01", 10, 2, 2, "FF", "Z6"),
            ("2025-06-01", 10, 1, 2, "SL", "Z2"),
            ("2025-06-02", 20, 1, 1, "CU", "Z8"),
            ("2025-06-01", 10, 2, 1, "SI", "Z5"),
            ("2025-06-01", 10, 1, 1, "FF", "Z1"),
        ],
        columns=["game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type", "zone_token"],
    )

    histories = list(iter_pitch_histories(frame, "pitch_type"))

    assert [(row["game_pk"], row["at_bat_number"], row["pitch_number"]) for row, _ in histories] == [
        (10, 1, 2), (10, 2, 2), (20, 1, 2),
    ]
    assert [history["seq_pitch_type"] for _, history in histories] == [
        "PAD|PAD|PAD|PAD|FF", "PAD|PAD|PAD|PAD|SI", "PAD|PAD|PAD|PAD|CU",
    ]
    assert [history["seq_zone"] for _, history in histories] == [
        "ZPAD|ZPAD|ZPAD|ZPAD|Z1", "ZPAD|ZPAD|ZPAD|ZPAD|Z5", "ZPAD|ZPAD|ZPAD|ZPAD|Z8",
    ]
    assert [history["seq_len"] for _, history in histories] == [1, 1, 1]
    assert [(history["prev_pitch_type"], history["prev_zone"]) for _, history in histories] == [
        ("FF", "Z1"), ("SI", "Z5"), ("CU", "Z8"),
    ]
