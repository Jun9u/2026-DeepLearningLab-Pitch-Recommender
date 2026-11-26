"""Generate fictional pitch records for tests and the standalone demo."""

EXAMPLE_PITCHER = 900001
DEMO_MARKER = "synthetic-demo"


def pitch(at_bat, pitch_number, pitch_type, **fields):
    return {
        "game_date": "2025-06-01",
        "game_pk": 100,
        "game_type": "R",
        "at_bat_number": at_bat,
        "pitch_number": pitch_number,
        "pitcher": EXAMPLE_PITCHER,
        "batter": 900101,
        "pitch_type": pitch_type,
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "inning": 1,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "p_throws": "R",
        "stand": "L",
        "zone": 1,
        "plate_x": 0.1,
        "plate_z": 2.5,
        "description": "called_strike",
        "events": None,
        "delta_run_exp": 0.0,
        "bat_score": 0,
        "fld_score": 0,
        "bat_score_diff": 0,
    } | fields


def pipeline_pitches(zones=tuple(range(1, 10))):
    pitch_types = ("FF", "SL", "CH")
    return [
        pitch(
            at_bat % 100 + 1, index + 1, pitch_types[(at_bat + index) % 3],
            game_date=f"2025-06-{1 + at_bat // 100:02d}",
            game_pk=100 + at_bat // 100,
            pitcher=EXAMPLE_PITCHER + at_bat % 4,
            batter=900101 + at_bat % 30,
            balls=index // 2,
            strikes=index % 2,
            outs_when_up=at_bat % 3,
            inning=1 + at_bat % 9,
            on_1b=900201 if at_bat % 3 else None,
            on_2b=900202 if at_bat % 5 == 0 else None,
            on_3b=900203 if at_bat % 7 == 0 else None,
            p_throws="R" if at_bat % 2 else "L",
            stand="L" if at_bat % 2 else "R",
            zone=zones[(at_bat // 3 + index) % len(zones)],
            events="field_out" if index == 3 else None,
            delta_run_exp=((at_bat + index) % 3 - 1) * 0.05 + (at_bat % 11 - 5) * 0.01,
        )
        for at_bat in range(900)
        for index in range(4)
    ]
