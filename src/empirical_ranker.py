"""Shared aggregation and scoring for the empirical outcome ranker."""

TARGET_COL = "delta_run_exp"
DEFAULT_ALPHA = 50.0
DEFAULT_THRESHOLDS = {
    "pitcher_count_action_zone": 5,
    "pitcher_action_zone": 10,
    "count_action_zone": 20,
    "action_zone": 30,
    "action": 50,
}

STAT_GROUPS = {
    "pitcher_count_action_zone": ["pitcher", "balls", "strikes", "action_pitch_type", "action_zone"],
    "pitcher_action_zone": ["pitcher", "action_pitch_type", "action_zone"],
    "count_action_zone": ["balls", "strikes", "action_pitch_type", "action_zone"],
    "action_zone": ["action_pitch_type", "action_zone"],
    "action": ["action_pitch_type"],
}


def zone_sort_key(zone):
    try:
        return int(str(zone).replace("Z", ""))
    except ValueError:
        return 999


def candidate_zones(train_df):
    return sorted(
        [
            zone for zone in train_df["action_zone"].dropna().unique().tolist()
            if str(zone).startswith("Z") and zone not in ["ZPAD", "ZUNK"]
        ],
        key=zone_sort_key,
    )


def top_pitch_types(pitches):
    """Select the three most frequent recommendable pitch types."""
    return [pitch for pitch in pitches.value_counts().index if pitch != "OTHER"][:3]


def select_candidate_pitches(lstm_pitches, pitcher_pitches, global_pitches, supported):
    """Keep LSTM then pitcher order, with global pitches as the last fallback."""
    for pitches in ((*lstm_pitches, *pitcher_pitches), global_pitches):
        normalized = (str(pitch).strip().upper() for pitch in pitches)
        candidates = list(dict.fromkeys(pitch for pitch in normalized if pitch in supported))
        if candidates:
            return candidates
    return []


def build_stat_tables(train_df, *, normalize_key_value=None):
    """Aggregate each tier, optionally normalizing keys for artifact storage."""
    tables = {}
    for table, columns in STAT_GROUPS.items():
        grouped = (
            train_df.groupby(columns, dropna=False)[TARGET_COL]
            .agg(["sum", "count", "mean"])
            .reset_index()
        )
        entries = {}
        for _, row in grouped.iterrows():
            values = tuple(row[column] for column in columns)
            key = values if normalize_key_value is None else tuple(map(normalize_key_value, values))
            entries[key] = {
                "sum": float(row["sum"]),
                "count": int(row["count"]),
                "mean": float(row["mean"]),
            }
        tables[table] = entries
    return tables


def smoothed_mean(sum_value, count_value, global_mean, alpha=DEFAULT_ALPHA):
    return (sum_value + alpha * global_mean) / (count_value + alpha)


def score_action(
    pitcher, balls, strikes, pitch_type, zone, *, stats, thresholds, global_mean, alpha, global_count,
):
    """Use the most specific tier with enough samples. Lower scores favor the pitcher."""
    tiers = (
        ("pitcher_count_action_zone", (pitcher, balls, strikes, pitch_type, zone)),
        ("pitcher_action_zone", (pitcher, pitch_type, zone)),
        ("count_action_zone", (balls, strikes, pitch_type, zone)),
        ("action_zone", (pitch_type, zone)),
        ("action", (pitch_type,)),
    )
    for table, key in tiers:
        stat = stats[table].get(key)
        if stat is not None and stat["count"] >= thresholds[table]:
            score = smoothed_mean(stat["sum"], stat["count"], global_mean, alpha)
            return score, table.replace("_", "+"), stat["count"]

    return global_mean, "global", global_count
