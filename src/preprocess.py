import pandas as pd

if __package__:
    from .raw_statcast import MODEL_CSV, raw_statcast_directory
    from .run_paths import run_directory
    from .sequence_helpers import add_runner_flags, sort_pitches
else:
    from raw_statcast import MODEL_CSV, raw_statcast_directory
    from run_paths import run_directory
    from sequence_helpers import add_runner_flags, sort_pitches

RUN_DIR = run_directory()
OUT_PATH = RUN_DIR / "data/processed/clean_pitch_data.csv"

USE_COLS = [
    "pitch_type",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "stand",
    "p_throws",
    "zone",
    "description",
    "events",
]

def main():
    raw_path = raw_statcast_directory(RUN_DIR / "data/raw") / MODEL_CSV
    df = pd.read_csv(raw_path, usecols=USE_COLS)

    print("원본 크기:", df.shape)

    # 정답 없는 행 제거
    df = df.dropna(subset=["pitch_type"])

    # 날짜 변환
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # 주자 상황: 비어 있으면 주자 없음, 값이 있으면 주자 있음
    add_runner_flags(df)

    # 원래 주자 ID 컬럼 제거
    df = df.drop(columns=["on_1b", "on_2b", "on_3b"])

    # 정렬
    df = sort_pitches(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("전처리 후 크기:", df.shape)
    print("저장 완료:", OUT_PATH)

if __name__ == "__main__":
    main()
