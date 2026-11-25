import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import export_serving_artifacts as exporter
from src.serving_paths import CURRENT, GENERATIONS, serving_artifact_paths
from tests.support import write_csv


def test_export_restores_legacy_scaler_in_feature_order(tmp_path, monkeypatch):
    # Supply the numeric columns in the reverse of the checkpoint order.
    columns = ["runner_3b", "runner_2b", "runner_1b", "inning", "outs_when_up", "strikes", "balls"]
    numeric_rows = [
        [1, 0, 1, 10, 0, 2, 0],
        [0, 1, 0, 5, 2, 1, 3],
        [1, 0, 1, 8, 1, 2, 0],
        [0, 1, 1, 3, 2, 0, 1],
    ]
    order = [4, 1, 3, 2]
    rows = [
        dict(zip(columns, numeric)) | {
            "game_date": f"2025-06-{i:02d}", "game_pk": 100 + i, "at_bat_number": 1, "pitch_number": 2,
        }
        for i, numeric in zip(order, numeric_rows, strict=True)
    ]
    write_csv(tmp_path, "sequence.csv", rows)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    serving_path = tmp_path / "serving.pt"
    checkpoint = dict.fromkeys((
        "model_state_dict", "target_to_idx", "idx_to_target", "pitch_vocab", "zone_vocab",
        "pitcher_vocab", "stand_vocab", "throws_vocab",
    ))
    torch = SimpleNamespace(load=Mock(return_value=checkpoint), save=Mock())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(exporter, "SEQUENCE_PATH", tmp_path / "sequence.csv")
    monkeypatch.setattr(exporter, "LSTM_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(exporter, "LSTM_SERVING_PATH", serving_path)
    monkeypatch.delenv("PITCH_GAME_SPLIT", raising=False)

    exporter.export_lstm_serving_artifact()

    torch.save.assert_called_once()
    serving_artifact, destination = torch.save.call_args.args
    assert destination == serving_path
    assert serving_artifact["numeric_cols"] == columns[::-1]
    assert serving_artifact["scaler_mean"] == [2, 0.5, 2, 4, 0.5, 1, 0]
    assert serving_artifact["scaler_scale"] == [1, 0.5, 1, 1, 0.5, 1, 1]


def test_failed_export_keeps_the_previously_published_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "RUN_DIR", tmp_path)

    def write_classifier(path):
        path.parent.mkdir()
        path.write_bytes(b"classifier")

    monkeypatch.setattr(exporter, "export_lstm_serving_artifact", write_classifier)
    monkeypatch.setattr(exporter, "export_outcome_ranker_artifact", lambda path: path.write_bytes(b"new ranker"))
    monkeypatch.setattr(exporter, "safe_torch_load", Mock())
    monkeypatch.setitem(sys.modules, "src.train_lstm", SimpleNamespace(load_serving_model=Mock()))
    models = tmp_path / "models"

    exporter.main()
    previous_pair = serving_artifact_paths(models)
    generation = (models / CURRENT).read_text()
    saved_bytes = tuple(path.read_bytes() for path in previous_pair)
    assert saved_bytes == (b"classifier", b"new ranker")

    def fail(path):
        path.write_bytes(b"incomplete ranker")
        raise OSError("ranker export failed")

    monkeypatch.setattr(exporter, "export_outcome_ranker_artifact", fail)
    with pytest.raises(OSError, match="ranker export failed"):
        exporter.main()

    assert serving_artifact_paths(models) == previous_pair
    assert (models / CURRENT).read_text() == generation
    assert tuple(path.read_bytes() for path in previous_pair) == saved_bytes
    assert list((models / GENERATIONS).iterdir()) == [previous_pair[0].parent]
