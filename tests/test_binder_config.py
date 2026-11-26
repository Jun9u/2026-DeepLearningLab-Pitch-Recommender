from pathlib import Path
import runpy
from types import SimpleNamespace


def test_public_statcast_profiles_preserve_both_scorers():
    config_path = Path(__file__).resolve().parents[1] / "binder/jupyter_server_config.py"
    config = SimpleNamespace(ServerProxy=SimpleNamespace())
    runpy.run_path(str(config_path), init_globals={"c": config})

    servers = config.ServerProxy.servers
    default = servers["pitch-statcast"]
    legacy = servers["pitch-statcast-legacy"]
    assert default["environment"]["PITCH_RANKER_MODE"] == "count_zone"
    assert legacy["environment"]["PITCH_RANKER_MODE"] == "empirical"
    assert servers["pitch-statcast-count"] == default
    assert default["command"] == legacy["command"]
    assert Path(servers["pitch-demo"]["environment"]["PITCH_RUN_DIR"]).name == "synthetic-demo"
