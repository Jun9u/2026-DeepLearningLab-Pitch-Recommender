import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src import download_statcast as download
from src.raw_statcast import COLUMNS_FILE, CURRENT, GENERATIONS, MODEL_CSV, OUTCOME_CSV, raw_statcast_directory
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

    download.cli(argv)

    provider.assert_called_once_with(*expected)
    bundle = raw_statcast_directory(raw)
    assert bundle.parent == raw / GENERATIONS
    assert pd.read_csv(bundle / MODEL_CSV)["pitch_type"].tolist() == ["FF"]
    assert pd.read_csv(bundle / OUTCOME_CSV)["pitch_type"].tolist() == ["FF"]
    assert (bundle / COLUMNS_FILE).read_text().splitlines() == provider.return_value.columns.tolist()


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


@pytest.mark.parametrize("rows", [[], [pitch(1, 1, "FF", game_type="D")]])
def test_empty_regular_season_download_preserves_existing_files(rows, monkeypatch, tmp_path):
    provider = Mock(return_value=pd.DataFrame(rows))
    monkeypatch.setitem(sys.modules, "pybaseball", SimpleNamespace(statcast=provider, cache=Mock()))
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    originals = {tmp_path / name: name.encode() for name in (MODEL_CSV, OUTCOME_CSV, COLUMNS_FILE)}
    for path, data in originals.items():
        path.write_bytes(data)

    with pytest.raises(ValueError, match="정규시즌 투구가 없습니다"):
        download.main("2025-10-01", "2025-10-01")

    assert {path: path.read_bytes() for path in originals} == originals
    assert raw_statcast_directory(tmp_path) == tmp_path
    assert not (tmp_path / CURRENT).exists()


@pytest.mark.parametrize("failure", ["second_csv", "pointer"])
def test_failed_collection_keeps_the_previous_generation(failure, monkeypatch, tmp_path):
    provider = Mock(side_effect=(
        pd.DataFrame([pitch(1, 1, "FF")]),
        pd.DataFrame([pitch(1, 1, "SL")]),
        pd.DataFrame([pitch(1, 1, "CH")]),
    ))
    monkeypatch.setitem(sys.modules, "pybaseball", SimpleNamespace(statcast=provider, cache=Mock()))
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    download.main()
    previous = raw_statcast_directory(tmp_path)
    originals = {path: path.read_bytes() for path in previous.iterdir()}

    with monkeypatch.context() as scoped:
        if failure == "second_csv":
            to_csv = pd.DataFrame.to_csv

            def write_csv(frame, path, **kwargs):
                if path.name == OUTCOME_CSV:
                    assert raw_statcast_directory(tmp_path) == previous
                    raise OSError("simulated write failure")
                return to_csv(frame, path, **kwargs)

            scoped.setattr(pd.DataFrame, "to_csv", write_csv)
        else:
            replace = Path.replace

            def replace_path(path, target):
                if target == tmp_path / CURRENT:
                    raise OSError("simulated write failure")
                return replace(path, target)

            scoped.setattr(Path, "replace", replace_path)

        with pytest.raises(OSError, match="simulated write failure"):
            download.main()

    assert raw_statcast_directory(tmp_path) == previous
    assert list((tmp_path / GENERATIONS).iterdir()) == [previous]
    assert {path: path.read_bytes() for path in originals} == originals

    download.main()
    current = raw_statcast_directory(tmp_path)
    assert current != previous
    assert pd.read_csv(current / MODEL_CSV)["pitch_type"].tolist() == ["CH"]
    assert pd.read_csv(current / OUTCOME_CSV)["pitch_type"].tolist() == ["CH"]
    assert {path: path.read_bytes() for path in originals} == originals
