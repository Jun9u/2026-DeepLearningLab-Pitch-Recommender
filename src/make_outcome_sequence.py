import argparse
import os

import pandas as pd
import numpy as np

if __package__:
    from .game_split import split_indices
    from .run_paths import run_directory
    from .sequence_helpers import add_runner_flags, iter_pitch_histories, left_pad as left_pad
else:
    from game_split import split_indices
    from run_paths import run_directory
    from sequence_helpers import add_runner_flags, iter_pitch_histories, left_pad as left_pad


# =========================
# 설정
# =========================

RUN_DIR = run_directory()
INPUT_PATH = RUN_DIR / "data/raw/savant_data_with_outcome.csv"
OUT_PATH = RUN_DIR / "data/processed/outcome_sequence_data.csv"

MIN_PITCH_COUNT = 500
GROUP_RARE_PITCHES = True

USE_COLS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "p_throws",
    "stand",
    "zone",
    "plate_x",
    "plate_z",
    "description",
    "events",
    "delta_run_exp",
    "bat_score",
    "fld_score",
    "bat_score_diff",
]

FORCE_OTHER_PITCHES = {"CS", "EP", "FA", "FO", "KN", "PO", "SC", "SV", "UN"}

def zone_to_token(x):
    if pd.isna(x):
        return "ZUNK"

    try:
        return f"Z{int(float(x))}"
    except ValueError:
        return "ZUNK"


def main(*, train_only_rare_pitches=False):
    if train_only_rare_pitches and not os.environ.get("PITCH_GAME_SPLIT"):
        raise ValueError("--train-only-rare-pitches requires PITCH_GAME_SPLIT")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {INPUT_PATH}")

    cols = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()
    missing = [c for c in USE_COLS if c not in cols]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = pd.read_csv(INPUT_PATH, usecols=USE_COLS)

    print("===== 원본 크기 =====")
    print(df.shape)

    df["delta_run_exp"] = pd.to_numeric(df["delta_run_exp"], errors="coerce")

    # 기대득점 모델에서는 현재 pitch_type, 현재 zone, delta_run_exp가 모두 필요함
    df = df.dropna(subset=["pitch_type", "zone", "delta_run_exp"]).copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # 주자 상황 변환
    add_runner_flags(df)

    # zone 토큰화
    df["zone_token"] = df["zone"].apply(zone_to_token)

    # 이상한 zone은 현재 action으로 쓰기 어렵기 때문에 제외
    df = df[df["zone_token"] != "ZUNK"].copy()

    # 희귀 구종 통합
    df["pitch_type_original"] = df["pitch_type"].astype(str)
    counts = df["pitch_type_original"].value_counts()

    if GROUP_RARE_PITCHES and train_only_rare_pitches:
        train, _, _ = split_indices(df, with_validation=False)
        train_counts = df.iloc[train]["pitch_type_original"].value_counts()
        frequent = set(train_counts[train_counts >= MIN_PITCH_COUNT].index)
        rare_set = set(counts.index) - frequent
        print("\n===== 랭커 적합 경기의 pitch_type 분포 =====")
        print(train_counts.head(30))
    elif GROUP_RARE_PITCHES:
        rare_set = set(counts[counts < MIN_PITCH_COUNT].index)
    else:
        rare_set = set()

    rare_set = rare_set.union(FORCE_OTHER_PITCHES)

    df["pitch_type_mapped"] = df["pitch_type_original"].apply(
        lambda x: "OTHER" if x in rare_set else x
    )

    print("\n===== 희귀 구종 처리 전 pitch_type 분포 =====")
    print(counts.head(30))

    print("\n===== OTHER로 묶을 희귀 구종 =====")
    print(sorted(rare_set))

    print("\n===== 희귀 구종 처리 후 pitch_type 분포 =====")
    print(df["pitch_type_mapped"].value_counts())

    samples = []
    for current, history in iter_pitch_histories(df, "pitch_type_mapped"):
        sample = {
            "game_date": current["game_date"],
            "game_pk": current["game_pk"],
            "at_bat_number": current["at_bat_number"],
            "pitch_number": current["pitch_number"],

            # state: 이전 흐름
            **history,

            # state: 현재 경기 상황
            "balls": current["balls"],
            "strikes": current["strikes"],
            "outs_when_up": current["outs_when_up"],
            "inning": current["inning"],
            "runner_1b": current["runner_1b"],
            "runner_2b": current["runner_2b"],
            "runner_3b": current["runner_3b"],

            "bat_score": current.get("bat_score", np.nan),
            "fld_score": current.get("fld_score", np.nan),
            "bat_score_diff": current.get("bat_score_diff", np.nan),

            "pitcher": current["pitcher"],
            "batter": current["batter"],
            "stand": current["stand"],
            "p_throws": current["p_throws"],

            # action: 실제 선택된 구종 + 실제 도착 zone
            "action_pitch_type": current["pitch_type_mapped"],
            "action_pitch_type_original": current["pitch_type_original"],
            "action_zone": current["zone_token"],
            "action_zone_raw": current["zone"],
            "plate_x": current["plate_x"],
            "plate_z": current["plate_z"],

            # target
            "delta_run_exp": current["delta_run_exp"],

            # 분석용
            "description": current["description"],
            "events": current["events"],
        }

        samples.append(sample)

    seq_df = pd.DataFrame(samples)

    print("\n===== outcome sequence 크기 =====")
    print(seq_df.shape)

    print("\n===== action_pitch_type 분포 =====")
    print(seq_df["action_pitch_type"].value_counts())

    print("\n===== action_zone 분포 =====")
    print(seq_df["action_zone"].value_counts().sort_index())

    print("\n===== delta_run_exp 요약 =====")
    print(seq_df["delta_run_exp"].describe())
    print("\nquantile:")
    print(seq_df["delta_run_exp"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n저장 완료:", OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-only-rare-pitches", action="store_true",
        help="Count rare pitches only in the games used to fit the outcome ranker (requires PITCH_GAME_SPLIT).",
    )
    main(train_only_rare_pitches=parser.parse_args().train_only_rare_pitches)
