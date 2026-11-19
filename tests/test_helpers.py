"""Small tests that run without downloaded Statcast data or model artifacts."""

import math

from src.make_sequence import left_pad, map_rare_token_sequence, zone_to_token
from src.outcome_empirical_recommender import (
    parse_top3,
    smoothed_mean,
    zone_sort_key,
)


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
