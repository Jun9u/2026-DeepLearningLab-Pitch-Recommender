import sys
from pathlib import Path

c.ServerProxy.servers = {
    "pitch-demo": {
        "command": [
            sys.executable, "-m", "uvicorn", "web.backend.app:app",
            "--host", "127.0.0.1", "--port", "{port}",
        ],
        "environment": {
            "PITCH_RUN_DIR": str(Path.home() / "runs/synthetic-demo"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        },
        "timeout": 60,
    },
}
