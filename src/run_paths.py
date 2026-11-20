"""Resolve the common directory for pipeline inputs and generated artifacts."""
import os
from pathlib import Path


def run_directory(default: Path | None = None) -> Path:
    """An override is relative to the process working directory unless absolute."""
    configured = os.environ.get("PITCH_RUN_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() if default is None else default
