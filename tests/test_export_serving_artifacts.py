import sys
from types import SimpleNamespace
from unittest.mock import Mock

from src import export_serving_artifacts as exporter
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
