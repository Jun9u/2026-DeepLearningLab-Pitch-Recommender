import argparse
from datetime import date
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd

if __package__:
    from .raw_statcast import COLUMNS_FILE, CURRENT, GENERATIONS, MODEL_CSV, OUTCOME_CSV
    from .run_paths import run_directory
else:
    from raw_statcast import COLUMNS_FILE, CURRENT, GENERATIONS, MODEL_CSV, OUTCOME_CSV
    from run_paths import run_directory



# =========================
# 설정
# =========================

START_DATE = "2023-03-01"
END_DATE = "2025-09-27"

RUN_DIR = run_directory()
RAW_DIR = RUN_DIR / "data/raw"

# 현재 피치타입 예측 모델에 필요한 기본 컬럼
MODEL_COLS = [
    "game_date",
    "game_pk",
    "game_type",
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
]

# 추후 기대득점 기반 outcome model용 후보 컬럼
OUTCOME_COLS = [
    "delta_run_exp",
    "delta_home_win_exp",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "post_home_score",
    "post_away_score",
    "post_bat_score",
    "post_fld_score",
    "home_score_diff",
    "bat_score_diff",
    "home_win_exp",
    "bat_win_exp",
]

# 추후 더 정교한 모델에 쓸 수 있는 추가 컬럼
OPTIONAL_DETAIL_COLS = [
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "type",
    "des",
    "bb_type",
    "hit_location",
    "launch_speed",
    "launch_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
]

def print_col_status(cols, target_cols, title):
    print(f"\n===== {title} =====")
    for col in target_cols:
        print(f"{col}: {col in cols}")


def main(start_date=None, end_date=None):
    from pybaseball import cache, statcast

    start_date = START_DATE if start_date is None else start_date
    end_date = END_DATE if end_date is None else end_date

    cache.enable()
    print("===== Statcast 데이터 다운로드 =====")
    print("기간:", start_date, "~", end_date)

    df = statcast(start_date, end_date)

    if "game_type" in df.columns:
        print("\n===== game_type 분포 =====")
        print(df["game_type"].value_counts(dropna=False))

        # 정규시즌만 사용할 경우
        df = df[df["game_type"] == "R"].copy()

    if df.empty:
        raise ValueError(f"Statcast 정규시즌 투구가 없습니다 ({start_date}~{end_date}).")

    print("\n===== 정규시즌 필터링 후 크기 =====")
    print(df.shape)

    print("\n===== 원본 데이터 크기 =====")
    print(df.shape)

    cols = df.columns.tolist()

    print("\n===== 전체 컬럼 수 =====")
    print(len(cols))

    print_col_status(cols, MODEL_COLS, "현재 피치타입 예측 모델 컬럼 존재 여부")
    print_col_status(cols, OUTCOME_COLS, "기대득점/승리확률 관련 컬럼 존재 여부")
    print_col_status(cols, OPTIONAL_DETAIL_COLS, "추가 세부 컬럼 존재 여부")

    missing_model_cols = [c for c in MODEL_COLS if c not in cols]

    if missing_model_cols:
        raise ValueError(f"현재 모델 필수 컬럼 누락: {missing_model_cols}")

    model_df = df[MODEL_COLS].copy()
    wide_cols = []
    for col in MODEL_COLS + OUTCOME_COLS + OPTIONAL_DETAIL_COLS:
        if col in cols and col not in wide_cols:
            wide_cols.append(col)

    wide_df = df[wide_cols].copy()

    print("\n===== delta_run_exp 확인 =====")
    if "delta_run_exp" in df.columns:
        delta = pd.to_numeric(df["delta_run_exp"], errors="coerce")

        print("delta_run_exp 존재: True")
        print("결측치 수:", int(delta.isna().sum()))
        print("유효값 수:", int(delta.notna().sum()))
        print("min:", delta.min())
        print("max:", delta.max())
        print("mean:", delta.mean())
        print("std:", delta.std())
        print("quantile:")
        print(delta.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    else:
        print("delta_run_exp 존재: False")
        print("현재 pybaseball 결과에 delta_run_exp가 없습니다.")
        print("이 경우 기대득점 기반 최적 추천은 현재 데이터셋으로는 바로 구현하기 어렵습니다.")

    print("\n===== pitch_type 분포 =====")
    print(model_df["pitch_type"].value_counts(dropna=False).head(30))

    generations = RAW_DIR / GENERATIONS
    generations.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    published = generations / generation
    with TemporaryDirectory(prefix=".statcast-", dir=RAW_DIR) as temporary:
        staging = Path(temporary)
        bundle = staging / generation
        bundle.mkdir()
        model_df.to_csv(bundle / MODEL_CSV, index=False, encoding="utf-8-sig")
        wide_df.to_csv(bundle / OUTCOME_CSV, index=False, encoding="utf-8-sig")
        (bundle / COLUMNS_FILE).write_text("".join(f"{col}\n" for col in cols), encoding="utf-8")
        pointer = staging / CURRENT
        pointer.write_text(generation + "\n", encoding="utf-8")
        bundle.rename(published)
        try:
            pointer.replace(RAW_DIR / CURRENT)
        except OSError:
            shutil.rmtree(published)
            raise

    print("\n===== Statcast 수집 저장 완료 =====")
    print(published)
    print(MODEL_CSV, model_df.shape)
    print(OUTCOME_CSV, wide_df.shape)
    print(COLUMNS_FILE)


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Download regular-season Statcast pitches.")
    parser.add_argument("--start-date", type=date.fromisoformat, default=START_DATE, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", type=date.fromisoformat, default=END_DATE, metavar="YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    main(args.start_date.isoformat(), args.end_date.isoformat())


if __name__ == "__main__":
    cli()
