"""Build a stand-alone serving pair from fictional pitches."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .synthetic_pitches import DEMO_MARKER, pipeline_pitches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/synthetic-demo"))
    run_dir = parser.parse_args().run_dir.expanduser().resolve()
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        parser.error(f"Use a new run directory. Already exists: {run_dir}")

    rows = pd.DataFrame(pipeline_pitches(zones=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)))
    raw_dir = run_dir / "data/raw"
    raw_dir.mkdir(parents=True)
    for name in ("savant_data.csv", "savant_data_with_outcome.csv"):
        rows.to_csv(raw_dir / name, index=False)

    env = dict(os.environ, PITCH_RUN_DIR=str(run_dir), CUDA_VISIBLE_DEVICES="")
    env["PITCH_GAME_SPLIT"] = str(run_dir / "data/processed/game_split.csv")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "2")
    for module in ("game_split", "preprocess", "make_sequence", "train_lstm",
                   "make_outcome_sequence", "export_serving_artifacts"):
        subprocess.run([sys.executable, "-m", f"src.{module}"], check=True, env=env)

    (run_dir / DEMO_MARKER).touch()
    print(f"Synthetic demo ready: {run_dir}")


if __name__ == "__main__":
    main()
