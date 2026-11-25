import math
from pathlib import Path
import pickle
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd

if __package__:
    from . import empirical_ranker as ranker
    from .game_split import split_indices
    from .run_paths import run_directory
    from .serving_paths import CURRENT, GENERATIONS, LSTM_NAME, OUTCOME_NAME
    from .training_features import NUMERIC_COLS
else:
    import empirical_ranker as ranker
    from game_split import split_indices
    from run_paths import run_directory
    from serving_paths import CURRENT, GENERATIONS, LSTM_NAME, OUTCOME_NAME
    from training_features import NUMERIC_COLS


# =========================
# 경로 설정
# =========================

RUN_DIR = run_directory()
LSTM_CHECKPOINT_PATH = RUN_DIR / "models/lstm_weighted_macro_best.pt"
LSTM_SERVING_PATH = RUN_DIR / "models" / LSTM_NAME

OUTCOME_PATH = RUN_DIR / "data/processed/outcome_sequence_data.csv"
OUTCOME_ARTIFACT_PATH = RUN_DIR / "models" / OUTCOME_NAME

SEQUENCE_PATH = RUN_DIR / "data/processed/sequence_data.csv"


def safe_torch_load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def normalize_key_value(x):
    if pd.isna(x):
        return None

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (np.floating, float)):
        if float(x).is_integer():
            return int(x)
        return float(x)

    return str(x)


def compute_scaler_from_sequence_data():
    if not SEQUENCE_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {SEQUENCE_PATH}")

    usecols = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
    ] + NUMERIC_COLS

    df = pd.read_csv(SEQUENCE_PATH, usecols=usecols)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    train, _, _ = split_indices(df)
    train_df = df.iloc[train].copy()

    numeric = train_df[NUMERIC_COLS].astype(float).values

    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)

    # StandardScaler와 동일하게 0분산 방지
    scale[scale == 0] = 1.0

    return mean.astype(float).tolist(), scale.astype(float).tolist()


def export_lstm_serving_artifact(destination=None):
    import torch

    if not LSTM_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {LSTM_CHECKPOINT_PATH}")

    checkpoint = safe_torch_load(LSTM_CHECKPOINT_PATH)

    # 예전 checkpoint에는 scaler가 없으므로 sequence_data에서 복원
    if "scaler_mean" not in checkpoint or "scaler_scale" not in checkpoint:
        print("scaler_mean/scaler_scale이 checkpoint에 없습니다.")
        print("sequence_data.csv의 train 70% 구간에서 scaler 값을 재계산합니다.")

        scaler_mean, scaler_scale = compute_scaler_from_sequence_data()

        checkpoint["scaler_mean"] = scaler_mean
        checkpoint["scaler_scale"] = scaler_scale
        checkpoint["numeric_cols"] = NUMERIC_COLS

    required_keys = [
        "model_state_dict",
        "target_to_idx",
        "idx_to_target",
        "pitch_vocab",
        "zone_vocab",
        "pitcher_vocab",
        "stand_vocab",
        "throws_vocab",
        "numeric_cols",
        "scaler_mean",
        "scaler_scale",
    ]

    missing = [k for k in required_keys if k not in checkpoint]

    if missing:
        raise ValueError(f"LSTM checkpoint에 서빙 필수 정보가 없습니다. 누락: {missing}")

    path = LSTM_SERVING_PATH if destination is None else destination
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

    print("===== LSTM serving artifact 저장 완료 =====")
    print(path)


def export_outcome_ranker_artifact(destination=None):
    if not OUTCOME_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {OUTCOME_PATH}")

    df = pd.read_csv(OUTCOME_PATH)

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    train, _, _ = split_indices(df, with_validation=False)
    train_df = df.iloc[train].copy()

    global_mean = float(train_df[ranker.TARGET_COL].mean())
    if not math.isfinite(global_mean):
        raise ValueError("랭커 적합 구간의 기대득점 평균을 계산할 수 없습니다.")

    print("\n===== Outcome artifact 생성 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("global_mean:", global_mean)

    stats = ranker.build_stat_tables(train_df, normalize_key_value=normalize_key_value)

    valid_zones = ranker.candidate_zones(train_df)

    pitcher_top_pitches = (
        train_df
        .groupby("pitcher")["action_pitch_type"]
        .agg(ranker.top_pitch_types)
        .to_dict()
    )

    # key를 int로 정리
    pitcher_top_pitches = {
        int(k): v for k, v in pitcher_top_pitches.items()
    }

    global_top_pitches = ranker.top_pitch_types(train_df["action_pitch_type"])

    artifact = {
        "target_col": ranker.TARGET_COL,
        "global_mean": global_mean,
        "alpha": ranker.DEFAULT_ALPHA,
        "thresholds": dict(ranker.DEFAULT_THRESHOLDS),
        "valid_zones": valid_zones,
        "pitcher_top_pitches": pitcher_top_pitches,
        "global_top_pitches": global_top_pitches,
        "stats": stats,
    }

    path = OUTCOME_ARTIFACT_PATH if destination is None else destination
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(artifact, f)

    print("\n===== Outcome ranker artifact 저장 완료 =====")
    print(path)
    print("valid_zones:", valid_zones)
    print("global_top_pitches:", global_top_pitches)

    print("\n통계 테이블 크기:")
    for name, value in artifact["stats"].items():
        print(name, len(value))


def main():
    models_dir = RUN_DIR / "models"
    generations = models_dir / GENERATIONS
    generations.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    published = generations / generation

    with TemporaryDirectory(prefix=".serving-", dir=models_dir) as temporary:
        staging = Path(temporary)
        bundle = staging / "bundle"
        lstm = bundle / LSTM_NAME
        export_lstm_serving_artifact(lstm)
        export_outcome_ranker_artifact(bundle / OUTCOME_NAME)

        if __package__:
            from .train_lstm import load_serving_model
        else:
            from train_lstm import load_serving_model

        load_serving_model(safe_torch_load(lstm), "cpu")
        pointer = staging / CURRENT
        pointer.write_text(generation + "\n", encoding="utf-8")
        bundle.rename(published)
        try:
            pointer.replace(models_dir / CURRENT)
        except OSError:
            shutil.rmtree(published)
            raise

    print("===== 서빙 모델 쌍 게시 완료 =====")
    print(published)


if __name__ == "__main__":
    main()
