"""Resolve the files from one completed Statcast collection."""

from pathlib import Path
from uuid import UUID


MODEL_CSV = "savant_data.csv"
OUTCOME_CSV = "savant_data_with_outcome.csv"
COLUMNS_FILE = "statcast_columns.txt"
CURRENT = "statcast-current"
GENERATIONS = "statcast"


def raw_statcast_directory(raw_dir: Path) -> Path:
    pointer = raw_dir / CURRENT
    try:
        generation = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return raw_dir

    try:
        if UUID(generation).hex != generation:
            raise ValueError
    except ValueError:
        raise ValueError(f"잘못된 Statcast 수집 식별자: {pointer}") from None

    return raw_dir / GENERATIONS / generation
