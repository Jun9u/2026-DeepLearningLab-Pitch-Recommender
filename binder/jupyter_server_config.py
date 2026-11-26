import sys
from pathlib import Path

cpu_environment = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
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
    "pitch-statcast": {
        "command": ["bash", str(Path.home() / "binder/statcast-demo"), "{port}"],
        "environment": cpu_environment,
        "timeout": 90,
    },
}
