import sys
from pathlib import Path

cpu_environment = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}
empirical_statcast_demo = {
    "command": ["bash", str(Path.home() / "binder/statcast-demo"), "{port}"],
    "environment": {**cpu_environment, "PITCH_RANKER_MODE": "empirical"},
    "timeout": 90,
}
count_zone_statcast_demo = {
    **empirical_statcast_demo,
    "environment": {**cpu_environment, "PITCH_RANKER_MODE": "count_zone"},
}

c.ServerProxy.servers = {
    "pitch-demo": {
        "command": [
            sys.executable, "-m", "uvicorn", "web.backend.app:app",
            "--host", "127.0.0.1", "--port", "{port}",
        ],
        "environment": {
            "PITCH_RUN_DIR": str(Path.home() / "runs/synthetic-demo"),
            **cpu_environment,
        },
        "timeout": 60,
    },
    "pitch-statcast": count_zone_statcast_demo,
    "pitch-statcast-count": count_zone_statcast_demo,
    "pitch-statcast-legacy": empirical_statcast_demo,
}
