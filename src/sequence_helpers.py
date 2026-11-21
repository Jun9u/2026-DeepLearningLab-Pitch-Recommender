"""Build pitch histories from rows selected by each preprocessing pipeline."""

PITCH_ORDER = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
AT_BAT_KEY = ["game_pk", "at_bat_number"]
HISTORY_LENGTH = 5
PITCH_PAD = "PAD"
ZONE_PAD = "ZPAD"


def sort_pitches(frame):
    return frame.sort_values(PITCH_ORDER).reset_index(drop=True)


def add_runner_flags(frame):
    """Add base occupancy without dropping the original runner IDs."""
    for base in ("1b", "2b", "3b"):
        frame[f"runner_{base}"] = frame[f"on_{base}"].notna().astype(int)


def left_pad(values, n, pad_token):
    values = list(values)[-n:]
    return [pad_token] * (n - len(values)) + values


def iter_pitch_histories(frame, pitch_column):
    """Yield each pitch with its padded history, after the first selected row of each at-bat."""
    for _, at_bat in sort_pitches(frame).groupby(AT_BAT_KEY, sort=False):
        at_bat = at_bat.sort_values("pitch_number").reset_index(drop=True)
        for index in range(1, len(at_bat)):
            previous = at_bat.iloc[max(0, index - HISTORY_LENGTH):index]
            pitches = left_pad(previous[pitch_column], HISTORY_LENGTH, PITCH_PAD)
            zones = left_pad(previous["zone_token"], HISTORY_LENGTH, ZONE_PAD)
            yield at_bat.iloc[index], {
                "seq_pitch_type": "|".join(pitches),
                "seq_zone": "|".join(zones),
                "seq_len": len(previous),
                "prev_pitch_type": pitches[-1],
                "prev_zone": zones[-1],
            }
