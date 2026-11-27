from hashlib import sha256
import json
import pickle

import pandas as pd
import pytest

from src.empirical_ranker import DEFAULT_ALPHA, DEFAULT_THRESHOLDS, STAT_GROUPS
from src.evaluate_serving_ranker import evaluate_release, main
from src.serving_paths import OUTCOME_NAME


@pytest.fixture
def evaluation_inputs(tmp_path):
    release = tmp_path / "release"
    model = release / "models" / OUTCOME_NAME
    model.parent.mkdir(parents=True)
    stats = {group: {} for group in STAT_GROUPS}
    stats["action"] = {("FF",): {"sum": 5.0, "count": 25, "mean": 0.2}}
    stats["action_zone"] = {
        ("FF", "Z1"): {"sum": 5.0, "count": 25, "mean": 0.2},
        ("OTHER", "Z2"): {"sum": 2.0, "count": 4, "mean": 0.5},
    }
    stats["count_action_zone"][(1, 1, "FF", "Z1")] = {"sum": 10.0, "count": 25, "mean": 0.4}
    model.write_bytes(pickle.dumps({
        "stats": stats, "global_mean": 0.1, "thresholds": DEFAULT_THRESHOLDS, "alpha": DEFAULT_ALPHA,
    }))
    manifest = {
        "ranker": {"fit_partitions": ["train", "validation"], "fit_rows": 100},
        "split": {"partitions": {
            "train": {"last_game_date": "2025-04-13"},
            "validation": {"last_game_date": "2025-06-07"},
            "test": {"last_game_date": "2025-09-27"},
        }},
        "artifacts": {f"models/{OUTCOME_NAME}": {"sha256": sha256(model.read_bytes()).hexdigest()}},
    }
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sequence = tmp_path / "sequence.csv"
    first = {
        "game_date": "2026-03-25", "game_pk": 100, "at_bat_number": 1, "pitch_number": 2,
        "pitcher": 10, "balls": 1, "strikes": 1, "action_pitch_type_original": "FF", "action_zone": "Z1",
        "action_pitch_type": "FF", "delta_run_exp": 0.4,
    }
    second = {**first, "pitch_number": 3, "action_pitch_type_original": "EP", "action_zone": "Z2",
              "delta_run_exp": 0.5}
    pd.DataFrame([first, second]).to_csv(sequence, index=False)
    return sequence, release, model


def test_serving_ranker_command_scores_fixed_artifact(evaluation_inputs, capsys):
    sequence, release, model = evaluation_inputs
    main(["--sequence-csv", str(sequence), "--release-dir", str(release)])
    report = json.loads(capsys.readouterr().out)
    assert report["fit_last_date"] == "2025-06-07"
    assert report["first_game_date"] == report["last_game_date"] == "2026-03-25"
    assert report["pitches"] == 2
    assert report["games"] == 1
    assert report["mapped_to_other"] == 1
    assert report["ranker_sha256"] == sha256(model.read_bytes()).hexdigest()
    assert report["sequence_sha256"] == sha256(sequence.read_bytes()).hexdigest()
    assert {name: result["mae"] for name, result in report["metrics"].items()} == pytest.approx({
        "empirical": 0.3, "count_zone": 0.05, "pitch_zone_mean": 0.1,
    })


def test_serving_ranker_rejects_pitches_in_fitting_period(evaluation_inputs):
    sequence, release, _ = evaluation_inputs
    rows = pd.read_csv(sequence)
    rows.loc[0, "game_date"] = "2025-06-07"
    rows.to_csv(sequence, index=False)
    with pytest.raises(ValueError, match="적합 종료일 2025-06-07"):
        evaluate_release(sequence, release)


def test_serving_ranker_rejects_artifact_with_wrong_hash(evaluation_inputs):
    sequence, release, model = evaluation_inputs
    model.write_bytes(model.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_release(sequence, release)
