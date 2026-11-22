import argparse
from collections import Counter
import json
import math
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__:
    from . import empirical_ranker as ranker
    from .run_paths import run_directory
else:
    import empirical_ranker as ranker
    from run_paths import run_directory


# =========================
# 설정
# =========================

RUN_DIR = run_directory()
DATA_PATH = RUN_DIR / "data/processed/outcome_sequence_data.csv"
LSTM_PRED_PATH = RUN_DIR / "results/lstm_weighted_macro_predictions.csv"

EVAL_OUTPUT_PATH = RUN_DIR / "results/outcome_empirical_eval.csv"
RECOMMEND_OUTPUT_PATH = RUN_DIR / "results/outcome_empirical_recommendations.csv"
DIAGNOSTICS_OUTPUT_PATH = RUN_DIR / "results/outcome_empirical_diagnostics.json"

TOP_N_RECOMMENDATIONS = 5

KEY_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
]


# =========================
# 유틸
# =========================

def load_lstm_predictions(test_df):
    prediction_path = LSTM_PRED_PATH
    if not prediction_path.exists():
        prediction_path = RUN_DIR / "outputs" / prediction_path.name

    if not prediction_path.exists():
        print("\nLSTM 예측 파일 없음. 투수별/전체 후보로 대체합니다.")
        test_df["top3_pitch_type"] = np.nan
        return test_df

    pred_cols = pd.read_csv(prediction_path, nrows=0).columns.tolist()

    needed = KEY_COLS + ["top3_pitch_type"]
    missing = [c for c in needed if c not in pred_cols]

    if missing:
        print("\nLSTM 예측 파일에 필요한 컬럼이 없습니다:", missing)
        test_df["top3_pitch_type"] = np.nan
        return test_df

    pred_df = pd.read_csv(prediction_path, usecols=needed)
    pred_df = pred_df.drop_duplicates(KEY_COLS)

    merged = test_df.merge(pred_df, on=KEY_COLS, how="left")

    print("\n===== LSTM top3 결합 =====")
    print("test rows:", len(test_df))
    print("merged rows:", len(merged))
    print("top3 결측:", merged["top3_pitch_type"].isna().sum())

    return merged


def parse_top3(text):
    if pd.isna(text):
        return []

    values = str(text).split("|")
    values = [v for v in values if v and v != "nan"]

    return values


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def write_diagnostics(y_true, ranker_metrics, *, train_count, global_mean, lstm_top3, actual_sources):
    baseline_metrics = regression_metrics(y_true, np.full(len(y_true), global_mean, dtype=float))
    matched = int(lstm_top3.notna().sum())
    report = {
        "target": ranker.TARGET_COL,
        "train_rows": train_count,
        "test_rows": len(y_true),
        "train_global_mean": float(global_mean),
        "observed_action_metrics": {
            name: {metric: float(value) if math.isfinite(value) else None for metric, value in metrics.items()}
            for name, metrics in (("empirical_ranker", ranker_metrics), ("train_global_mean", baseline_metrics))
        },
        "lstm_top3_join": {
            "matched_rows": matched,
            "missing_rows": len(y_true) - matched,
            "match_rate": matched / len(y_true),
        },
        "observed_action_source_counts": dict(Counter(actual_sources)),
    }
    DIAGNOSTICS_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print("진단 저장:", DIAGNOSTICS_OUTPUT_PATH)


def main(*, diagnostics=False):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # LSTM의 test 구간과 맞추기 위해 80/20 분할
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("===== 데이터 분할 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

    global_mean = train_df[ranker.TARGET_COL].mean()

    print("\n===== train delta_run_exp =====")
    print(train_df[ranker.TARGET_COL].describe())
    print("global_mean:", global_mean)

    # 계층적 통계 생성
    stats = ranker.build_stat_tables(train_df)

    print("\n===== 통계 테이블 크기 =====")
    print("pitcher+count+action+zone:", len(stats["pitcher_count_action_zone"]))
    print("pitcher+action+zone      :", len(stats["pitcher_action_zone"]))
    print("count+action+zone        :", len(stats["count_action_zone"]))
    print("action+zone              :", len(stats["action_zone"]))
    print("action                   :", len(stats["action"]))

    # 후보 zone
    valid_zones = ranker.candidate_zones(train_df)

    print("\n===== 후보 zone =====")
    print(valid_zones)

    # 투수별 후보 구종 fallback
    pitcher_top_pitches = (
        train_df
        .groupby("pitcher")["action_pitch_type"]
        .agg(lambda x: x.value_counts().index[:3].tolist())
        .to_dict()
    )

    global_top_pitches = train_df["action_pitch_type"].value_counts().index[:3].tolist()

    print("\n===== 전체 top pitches =====")
    print(global_top_pitches)

    train_count = len(train_df)

    def predict_delta(row, action_pitch_type, action_zone):
        return ranker.score_action(
            row["pitcher"], row["balls"], row["strikes"], action_pitch_type, action_zone,
            stats=stats, thresholds=ranker.DEFAULT_THRESHOLDS, global_mean=global_mean,
            alpha=ranker.DEFAULT_ALPHA, global_count=train_count,
        )

    # 실제 test action에 대한 예측 성능 확인
    actual_pred = []
    actual_sources = []

    for _, row in test_df.iterrows():
        pred, source, count = predict_delta(
            row,
            row["action_pitch_type"],
            row["action_zone"],
        )

        actual_pred.append(pred)
        actual_sources.append(source)

    y_true = test_df[ranker.TARGET_COL].astype(float).values
    y_pred = np.array(actual_pred)

    metrics = regression_metrics(y_true, y_pred)

    eval_df = pd.DataFrame(
        [
            {
                "model": "Empirical Outcome Ranker",
                "target": "delta_run_exp",
                **metrics,
                "global_mean": global_mean,
                "test_rows": len(test_df),
            }
        ]
    )

    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 실제 action delta_run_exp 예측 성능 =====")
    print(eval_df.to_string(index=False))
    print("저장:", EVAL_OUTPUT_PATH)

    # LSTM top3와 결합
    test_df = load_lstm_predictions(test_df)

    recommendation_rows = []

    for _, row in test_df.iterrows():
        lstm_candidates = parse_top3(row.get("top3_pitch_type", np.nan))

        if not lstm_candidates:
            lstm_candidates = pitcher_top_pitches.get(
                row["pitcher"],
                global_top_pitches,
            )

        # 중복 제거 + OTHER 제외
        candidate_pitches = []

        # 1. LSTM Top-3
        for p in lstm_candidates:
            p = str(p).strip()

            if p in ["", "nan", "None", "OTHER"]:
                continue

            if p not in candidate_pitches:
                candidate_pitches.append(p)

        # 2. 해당 투수의 자주 던지는 구종 Top-3 추가
        fallback_candidates = pitcher_top_pitches.get(row["pitcher"], [])

        for p in fallback_candidates:
            p = str(p).strip()

            if p in ["", "nan", "None", "OTHER"]:
                continue

            if p not in candidate_pitches:
                candidate_pitches.append(p)

        # 3. 그래도 비면 전체 top pitch 사용
        if not candidate_pitches:
            for p in global_top_pitches:
                p = str(p).strip()

                if p in ["", "nan", "None", "OTHER"]:
                    continue

                if p not in candidate_pitches:
                    candidate_pitches.append(p)

        candidates = []

        for pitch in candidate_pitches:
            for zone in valid_zones:
                score, source, count = predict_delta(row, pitch, zone)

                candidates.append(
                    {
                        "pitch_type": pitch,
                        "zone": zone,
                        "pred_delta_run_exp": score,
                        "source": source,
                        "source_count": count,
                    }
                )

        candidates = sorted(candidates, key=lambda x: x["pred_delta_run_exp"])
        top_recs = candidates[:TOP_N_RECOMMENDATIONS]

        out = {
            "game_date": row.get("game_date"),
            "game_pk": row.get("game_pk"),
            "at_bat_number": row.get("at_bat_number"),
            "pitch_number": row.get("pitch_number"),

            "pitcher": row.get("pitcher"),
            "batter": row.get("batter"),
            "balls": row.get("balls"),
            "strikes": row.get("strikes"),
            "outs_when_up": row.get("outs_when_up"),
            "inning": row.get("inning"),
            "runner_1b": row.get("runner_1b"),
            "runner_2b": row.get("runner_2b"),
            "runner_3b": row.get("runner_3b"),

            "seq_pitch_type": row.get("seq_pitch_type"),
            "seq_zone": row.get("seq_zone"),
            "lstm_top3_pitch_type": row.get("top3_pitch_type"),

            "actual_pitch_type": row.get("action_pitch_type"),
            "actual_zone": row.get("action_zone"),
            "actual_delta_run_exp": row.get("delta_run_exp"),
            "actual_description": row.get("description"),
            "actual_events": row.get("events"),
        }

        if top_recs:
            best = top_recs[0]

            out["recommended_pitch_type"] = best["pitch_type"]
            out["recommended_zone"] = best["zone"]
            out["recommended_pred_delta_run_exp"] = best["pred_delta_run_exp"]
            out["recommendation_source"] = best["source"]
            out["recommendation_source_count"] = best["source_count"]

            # 실제 선택의 예측 점수와 추천 선택의 예측 점수 차이
            actual_score, actual_source, actual_count = predict_delta(
                row,
                row["action_pitch_type"],
                row["action_zone"],
            )

            out["actual_action_pred_delta_run_exp"] = actual_score
            out["predicted_improvement_vs_actual_action"] = (
                actual_score - best["pred_delta_run_exp"]
            )

        for i, rec in enumerate(top_recs, start=1):
            out[f"rec_{i}"] = (
                f"{rec['pitch_type']}/{rec['zone']}:"
                f"{rec['pred_delta_run_exp']:.4f}"
                f"({rec['source']},n={rec['source_count']})"
            )

        recommendation_rows.append(out)

    rec_df = pd.DataFrame(recommendation_rows)
    rec_df.to_csv(RECOMMEND_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    if diagnostics:
        write_diagnostics(
            y_true, metrics, train_count=train_count, global_mean=global_mean,
            lstm_top3=test_df["top3_pitch_type"], actual_sources=actual_sources,
        )

    print("\n===== 추천 결과 샘플 =====")
    show_cols = [
        "lstm_top3_pitch_type",
        "recommended_pitch_type",
        "recommended_zone",
        "recommended_pred_delta_run_exp",
        "actual_pitch_type",
        "actual_zone",
        "actual_delta_run_exp",
        "predicted_improvement_vs_actual_action",
        "rec_1",
        "rec_2",
        "rec_3",
    ]

    show_cols = [c for c in show_cols if c in rec_df.columns]
    print(rec_df[show_cols].head(10).to_string(index=False))

    print("\n추천 결과 저장:", RECOMMEND_OUTPUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Write the observed-action ranker and training-mean comparison with the LSTM join rate.",
    )
    main(diagnostics=parser.parse_args().diagnostics)
