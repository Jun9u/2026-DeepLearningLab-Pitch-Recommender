import math

import pandas as pd
import pytest

from src import outcome_empirical_recommender as recommender
from src.empirical_ranker import (
    COUNT_ZONE_MODE, DEFAULT_THRESHOLDS, EMPIRICAL_MODE, build_stat_tables, candidate_zones, score_action,
    scoring_function,
)
from tests.support import write_csv


def test_score_action_uses_each_tier_at_its_minimum_count():
    tiers = [
        ("pitcher_count_action_zone", (123, 1, 2, "FF", "Z1"), 5),
        ("pitcher_action_zone", (123, "FF", "Z1"), 10),
        ("count_action_zone", (1, 2, "FF", "Z1"), 20),
        ("action_zone", ("FF", "Z1"), 30),
        ("action", ("FF",), 50),
    ]
    stats = {table: {key: {"sum": count * 0.5, "count": count}} for table, key, count in tiers}
    settings = {
        "stats": stats, "thresholds": DEFAULT_THRESHOLDS, "global_mean": 0.25,
        "alpha": 4.0, "global_count": 80,
    }

    for table, key, count in tiers:
        score, source, source_count = score_action(123, 1, 2, "FF", "Z1", **settings)
        assert math.isclose(score, (count * 0.5 + 1.0) / (count + 4.0))
        assert (source, source_count) == (table.replace("_", "+"), count)
        stats[table][key]["count"] -= 1

    assert score_action(123, 1, 2, "FF", "Z1", **settings) == (0.25, "global", 80)
    settings.update(global_count=0, thresholds={**DEFAULT_THRESHOLDS, "action": 49})
    assert score_action(123, 1, 2, "FF", "Z1", **settings)[1:] == ("action", 49)
    assert score_action(123, 1, 2, "CH", "Z1", **settings) == (0.25, "global", 0)


def test_count_zone_score_uses_pitch_zone_prior_and_fallbacks():
    score = scoring_function(COUNT_ZONE_MODE)
    assert scoring_function(EMPIRICAL_MODE) is score_action
    with pytest.raises(ValueError, match="Unsupported ranker mode"):
        scoring_function("typo")

    settings = {
        "stats": {
            "count_action_zone": {(1, 2, "FF", "Z1"): {"sum": 0.5, "count": 1}},
            "action_zone": {("FF", "Z1"): {"mean": 0.1, "count": 40}},
        },
        "global_mean": 0.25, "global_count": 80,
    }

    assert score(123, 1, 2, "FF", "Z1", **settings) == (3 / 26, "count+action+zone", 1)
    assert score(456, 1, 2, "FF", "Z1", **settings) == (3 / 26, "count+action+zone", 1)
    assert score(123, 0, 2, "FF", "Z1", **settings) == (0.1, "action+zone", 40)
    assert score(123, 1, 2, "FF", "Z2", **settings) == (0.25, "global", 80)

    settings["stats"]["count_action_zone"][(0, 2, "FF", "Z1")] = {"sum": 0.0, "count": 0}
    settings["stats"]["count_action_zone"][(1, 2, "CH", "Z1")] = {"sum": 0.0, "count": 0}
    settings["stats"]["action_zone"][("CH", "Z1")] = {"mean": math.nan, "count": 0}

    assert score(123, 0, 2, "FF", "Z1", **settings) == (0.1, "action+zone", 40)
    assert score(123, 1, 2, "CH", "Z1", **settings) == (0.25, "global", 80)


def test_stat_tables_preserve_raw_keys_and_accept_artifact_normalization():
    frame = pd.DataFrame(
        [(123, 1, 2, "FF", "Z11", 0.25), (123, 1, 2, "FF", "Z11", None), (123, 1, 2, "FF", None, 0.5)],
        columns=["pitcher", "balls", "strikes", "action_pitch_type", "action_zone", "delta_run_exp"],
    )
    raw = build_stat_tables(frame)["action_zone"]
    normalized = build_stat_tables(frame, normalize_key_value=lambda value: None if pd.isna(value) else value)

    assert raw[("FF", "Z11")] == {"sum": 0.25, "count": 1, "mean": 0.25}
    assert len(raw) == 2 and any(pd.isna(zone) for _, zone in raw)
    assert normalized["action_zone"][("FF", None)] == {"sum": 0.5, "count": 1, "mean": 0.5}
    assert normalized["action"][("FF",)] == {"sum": 0.75, "count": 2, "mean": 0.375}


def test_candidate_zones_are_sorted_numerically_and_exclude_padding():
    frame = pd.DataFrame({"action_zone": ["Z11", "ZUNK", "zone", "Z2", "ZPAD", "Z11", None]})
    assert candidate_zones(frame) == ["Z2", "Z11"]


def test_offline_ranker_scores_candidates_with_each_rows_count(monkeypatch, tmp_path):
    contexts = [(0, 2, "FF", "Z1", 0.0), (3, 1, "SL", "Z2", 0.4)]
    rows = [
        {
            "game_date": "2025-06-01",
            "game_pk": 100,
            "at_bat_number": index,
            "pitch_number": 2,
            "pitcher": 123,
            "balls": balls,
            "strikes": strikes,
            "action_pitch_type": pitch,
            "action_zone": zone,
            "delta_run_exp": outcome,
        }
        for index, (balls, strikes, pitch, zone, outcome) in enumerate(contexts * 10, start=1)
    ]
    write_csv(tmp_path, "outcomes.csv", rows)
    monkeypatch.delenv("PITCH_GAME_SPLIT", raising=False)
    monkeypatch.setattr(recommender, "RUN_DIR", tmp_path)
    monkeypatch.setattr(recommender, "DATA_PATH", tmp_path / "outcomes.csv")
    monkeypatch.setattr(recommender, "LSTM_PRED_PATH", tmp_path / "lstm.csv")
    monkeypatch.setattr(recommender, "EVAL_OUTPUT_PATH", tmp_path / "eval.csv")
    monkeypatch.setattr(recommender, "RECOMMEND_OUTPUT_PATH", tmp_path / "recommendations.csv")

    recommender.main()

    recommendations = pd.read_csv(recommender.RECOMMEND_OUTPUT_PATH)
    assert recommendations["recommendation_source"].tolist() == ["pitcher+count+action+zone", "global"] * 2
    assert recommendations["recommended_pitch_type"].tolist() == ["FF"] * 4
    assert recommendations["recommended_zone"].tolist() == ["Z1"] * 4
    actual_scores = recommendations["actual_action_pred_delta_run_exp"].tolist()
    assert all(math.isclose(actual, expected) for actual, expected in zip(actual_scores, [10 / 58, 13.2 / 58] * 2))
