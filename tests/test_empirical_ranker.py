import math

import pandas as pd

from src.empirical_ranker import DEFAULT_THRESHOLDS, build_stat_tables, candidate_zones, score_action


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
