import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src import download_statcast as download
from tests.support import pitch


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("2023-03-01", "2025-09-27")),
        (["--start-date", "2025-06-01", "--end-date", "2025-06-14"], ("2025-06-01", "2025-06-14")),
    ],
)
def test_cli_downloads_requested_range(argv, expected, monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    provider = Mock(return_value=pd.DataFrame([pitch(1, 1, "FF"), pitch(1, 2, "SL", game_type="S")]))
    monkeypatch.setitem(sys.modules, "pybaseball", SimpleNamespace(statcast=provider, cache=Mock()))
    monkeypatch.setattr(download, "RAW_DIR", raw)
    monkeypatch.setattr(download, "MODEL_OUTPUT_PATH", raw / "savant_data.csv")
    monkeypatch.setattr(download, "WIDE_OUTPUT_PATH", raw / "savant_data_with_outcome.csv")
    monkeypatch.setattr(download, "COLUMN_LIST_PATH", raw / "statcast_columns.txt")

    download.cli(argv)

    provider.assert_called_once_with(*expected)
    assert pd.read_csv(download.MODEL_OUTPUT_PATH)["pitch_type"].tolist() == ["FF"]
    assert pd.read_csv(download.WIDE_OUTPUT_PATH)["pitch_type"].tolist() == ["FF"]


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--start-date", "2025-02-30"], "2025-02-30"),
        (["--start-date", "2025-06-10", "--end-date", "2025-06-01"], "must be on or before"),
    ],
)
def test_cli_rejects_invalid_dates_before_downloading(argv, error, monkeypatch, capsys):
    download_data = Mock()
    monkeypatch.setattr(download, "main", download_data)

    with pytest.raises(SystemExit) as exit_info:
        download.cli(argv)

    assert exit_info.value.code == 2
    assert error in capsys.readouterr().err
    download_data.assert_not_called()
