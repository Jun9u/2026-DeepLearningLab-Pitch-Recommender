"""Run the original entry points without writing to the source checkout."""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PORT = re.compile(r"Uvicorn running on http://127\.0\.0\.1:(\d+)")


def pipeline_env(root):
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                PYBASEBALL_CACHE=str(root / "pybaseball-cache"), PYTHONIOENCODING="utf-8")


def command(root, *args):
    completed = subprocess.run(
        [sys.executable, *args], cwd=root, env=pipeline_env(root),
        capture_output=True, encoding="utf-8", timeout=120,
    )
    if completed.returncode:
        raise AssertionError(f"{args[0]} exited with {completed.returncode}\n{completed.stdout}{completed.stderr}")
    return completed.stdout


def run_step(root, module):
    return command(root, str(REPO / "src" / f"{module}.py"))


def stage_web(root):
    # The original app resolves artifact and UI paths relative to its own source.
    for directory in ("src", "web"):
        shutil.copytree(REPO / directory, root / directory, ignore=shutil.ignore_patterns("__pycache__"))


def uvicorn_args():
    return [sys.executable, "-m", "uvicorn", "web.backend.app:app", "--host", "127.0.0.1", "--port", "0"]


@contextmanager
def api_server(root):
    log_path = root / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            uvicorn_args(), cwd=root, env=pipeline_env(root), stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while process.poll() is None and time.monotonic() < deadline:
                output = log_path.read_text(encoding="utf-8")
                match = PORT.search(output)
                if match:
                    yield f"http://127.0.0.1:{match[1]}"
                    return
                time.sleep(0.1)
            raise AssertionError(log_path.read_text(encoding="utf-8"))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def write_csv(root, filename, rows):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def pitch(at_bat, pitch_number, pitch_type, **fields):
    return {
        "game_date": "2025-06-01",
        "game_pk": 100,
        "game_type": "R",
        "at_bat_number": at_bat,
        "pitch_number": pitch_number,
        "pitcher": 900001,
        "batter": 900101,
        "pitch_type": pitch_type,
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "inning": 1,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "p_throws": "R",
        "stand": "L",
        "zone": 1,
        "plate_x": 0.1,
        "plate_z": 2.5,
        "description": "called_strike",
        "events": None,
        "delta_run_exp": 0.0,
        "bat_score": 0,
        "fld_score": 0,
        "bat_score_diff": 0,
    } | fields


def pipeline_pitches():
    pitch_types = ("FF", "SL", "CH")
    return [
        pitch(
            at_bat % 100 + 1, index + 1, pitch_types[(at_bat + index) % 3],
            game_date=f"2025-06-{1 + at_bat // 100:02d}",
            game_pk=100 + at_bat // 100,
            pitcher=900001 + at_bat % 4,
            batter=900101 + at_bat % 30,
            balls=index // 2,
            strikes=index % 2,
            outs_when_up=at_bat % 3,
            inning=1 + at_bat % 9,
            on_1b=900201 if at_bat % 3 else None,
            on_2b=900202 if at_bat % 5 == 0 else None,
            on_3b=900203 if at_bat % 7 == 0 else None,
            p_throws="R" if at_bat % 2 else "L",
            stand="L" if at_bat % 2 else "R",
            zone=1 + (at_bat // 3 + index) % 9,
            events="field_out" if index == 3 else None,
            delta_run_exp=((at_bat + index) % 3 - 1) * 0.05 + (at_bat % 11 - 5) * 0.01,
        )
        for at_bat in range(900)
        for index in range(4)
    ]


def sparse_outcome_rows():
    # Four action/zone groups with three training observations each.
    actions = [("FF", "Z1"), ("FF", "Z2"), ("SL", "Z1"), ("SL", "Z2")]
    return [
        {
            "game_date": "2025-06-01",
            "game_pk": 100,
            "at_bat_number": index + 1,
            "pitch_number": 2,
            "pitcher": 900001 if index < 12 else 900002,
            "balls": 1,
            "strikes": 1,
            "action_pitch_type": action,
            "action_zone": zone,
            "delta_run_exp": 0.1 if index < 12 else (index - 12) / 10,
        }
        for index, (action, zone) in enumerate(actions * 3 + actions[:3])
    ]
