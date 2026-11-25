"""Resolve both serving artifacts from one published generation."""

from pathlib import Path
from uuid import UUID


LSTM_NAME = "lstm_serving_artifact.pt"
OUTCOME_NAME = "outcome_ranker_artifact.pkl"
CURRENT = "serving-current"
GENERATIONS = "serving"


def serving_artifact_paths(models_dir: Path) -> tuple[Path, Path]:
    try:
        generation = (models_dir / CURRENT).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return models_dir / LSTM_NAME, models_dir / OUTCOME_NAME

    try:
        if UUID(generation).hex != generation:
            raise ValueError
    except ValueError:
        raise ValueError(f"잘못된 모델 세대 식별자: {models_dir / CURRENT}") from None

    directory = models_dir / GENERATIONS / generation
    return directory / LSTM_NAME, directory / OUTCOME_NAME
