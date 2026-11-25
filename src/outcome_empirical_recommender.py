import argparse
from collections import Counter
from hashlib import file_digest
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__:
    from . import empirical_ranker as ranker
    from .game_split import GAME_SPLIT_HASH_COL, read_split_manifest, split_indices, split_manifest_hash
    from .run_paths import run_directory
else:
    import empirical_ranker as ranker
    from game_split import GAME_SPLIT_HASH_COL, read_split_manifest, split_indices, split_manifest_hash
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

def load_lstm_predictions(test_df, expected_split_hash):
    prediction_path = LSTM_PRED_PATH
    if not prediction_path.exists():
        prediction_path = RUN_DIR / "outputs" / prediction_path.name

    if not prediction_path.exists():
        print("\nLSTM 예측 파일 없음. 투수별/전체 후보로 대체합니다.")
        test_df["top3_pitch_type"] = np.nan
        return test_df

    needed = KEY_COLS + ["top3_pitch_type"]
    columns = {*needed, GAME_SPLIT_HASH_COL}
    pred_df = pd.read_csv(prediction_path, usecols=columns.__contains__)
    has_split_hash = GAME_SPLIT_HASH_COL in pred_df.columns
    if has_split_hash != (expected_split_hash is not None):
        raise ValueError(f"LSTM 예측의 경기 분할 정보가 현재 평가와 일치하지 않습니다: {prediction_path}")
    missing = [c for c in needed if c not in pred_df.columns]

    if missing:
        print("\nLSTM 예측 파일에 필요한 컬럼이 없습니다:", missing)
        test_df["top3_pitch_type"] = np.nan
        return test_df

    if has_split_hash:
        split_hashes = pred_df.pop(GAME_SPLIT_HASH_COL)
        if split_hashes.empty or not split_hashes.eq(expected_split_hash).fillna(False).all():
            raise ValueError(f"LSTM 예측의 경기 분할 SHA-256이 현재 평가와 일치하지 않습니다: {prediction_path}")
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


def write_diagnostics(
    output, test_df, y_true, ranker_metrics, *, train_count, global_mean, zone_stats, actual_sources,
    candidate_counts, result_paths,
):
    test_count = len(y_true)
    pair_predictions = []
    pair_matches = 0
    for key in zip(test_df["action_pitch_type"], test_df["action_zone"], strict=True):
        stat = zone_stats.get(key)
        if stat is not None and stat["count"] > 0:
            pair_predictions.append(stat["mean"])
            pair_matches += 1
        else:
            pair_predictions.append(global_mean)

    metrics_by_model = {
        "empirical_ranker": ranker_metrics,
        "train_global_mean": regression_metrics(y_true, np.full(test_count, global_mean, dtype=float)),
        "train_pitch_zone_mean": regression_metrics(y_true, pair_predictions),
    }
    matched = int(test_df["top3_pitch_type"].notna().sum())
    result_hashes = {}
    for path in result_paths:
        with path.open("rb") as stream:
            result_hashes[path.name] = file_digest(stream, "sha256").hexdigest()

    report = {
        "result_sha256": result_hashes,
        "target": ranker.TARGET_COL,
        "train_rows": train_count,
        "test_rows": test_count,
        "train_global_mean": float(global_mean),
        "observed_action_metrics": {
            name: {metric: float(value) if math.isfinite(value) else None for metric, value in metrics.items()}
            for name, metrics in metrics_by_model.items()
        },
        "train_pitch_zone_mean_counts": {
            "pair_mean_rows": pair_matches,
            "global_fallback_rows": test_count - pair_matches,
        },
        "candidate_coverage": {
            name: {
                "rows_with_candidates": counts["rows_with_candidates"],
                "observed_action_covered_rows": counts["observed_action_covered_rows"],
                "observed_action_coverage_rate": counts["observed_action_covered_rows"] / test_count,
                "average_candidates_per_row": counts["total_candidates"] / test_count,
            }
            for name, counts in candidate_counts.items()
        },
        "lstm_top3_join": {
            "matched_rows": matched,
            "missing_rows": test_count - matched,
            "match_rate": matched / test_count,
        },
        "observed_action_source_counts": dict(Counter(actual_sources)),
    }
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


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

    manifest = read_split_manifest()
    train, _, test = split_indices(df, with_validation=False, manifest_bytes=manifest)
    train_df = df.iloc[train].copy()
    test_df = df.iloc[test].copy()

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
        .agg(ranker.top_pitch_types)
        .to_dict()
    )

    global_top_pitches = ranker.top_pitch_types(train_df["action_pitch_type"])
    supported_pitches = {code for (code,) in stats["action"] if code != "OTHER"}

    print("\n===== 전체 top pitches =====")
    print(global_top_pitches)

    train_count = len(train_df)

    def predict_delta(context, action_pitch_type, action_zone):
        return ranker.score_action(
            *context, action_pitch_type, action_zone,
            stats=stats, thresholds=ranker.DEFAULT_THRESHOLDS, global_mean=global_mean,
            alpha=ranker.DEFAULT_ALPHA, global_count=train_count,
        )

    # 실제 test action에 대한 예측 성능 확인
    actual_pred = []
    actual_sources = []

    for _, row in test_df.iterrows():
        context = (row["pitcher"], row["balls"], row["strikes"])
        pred, source, count = predict_delta(
            context,
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

    # LSTM top3와 결합
    test_df = load_lstm_predictions(test_df, split_manifest_hash(manifest))

    recommendation_rows = []
    candidate_counts = {"lstm_top3_with_frequency": Counter(), "frequency_only": Counter()}
    zone_set = set(valid_zones)

    for _, row in test_df.iterrows():
        pitcher = row["pitcher"]
        context = (pitcher, row["balls"], row["strikes"])
        lstm_candidates = parse_top3(row.get("top3_pitch_type", np.nan))
        pitcher_pitches = pitcher_top_pitches.get(pitcher, [])
        candidate_pitches = ranker.select_candidate_pitches(
            lstm_candidates, pitcher_pitches, global_top_pitches, supported_pitches,
        )

        if diagnostics:
            frequency_pitches = ranker.select_candidate_pitches((), pitcher_pitches, global_top_pitches, supported_pitches)
            for rule, pitches in (("lstm_top3_with_frequency", candidate_pitches), ("frequency_only", frequency_pitches)):
                counts = candidate_counts[rule]
                size = len(pitches) * len(valid_zones)
                counts["total_candidates"] += size
                counts["rows_with_candidates"] += int(size > 0)
                counts["observed_action_covered_rows"] += int(
                    row["action_pitch_type"] in pitches and row["action_zone"] in zone_set
                )

        candidates = []

        for pitch in candidate_pitches:
            for zone in valid_zones:
                score, source, count = predict_delta(context, pitch, zone)

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
                context,
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
    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ranker-", dir=EVAL_OUTPUT_PATH.parent) as temporary:
        staging = Path(temporary)
        eval_csv = staging / EVAL_OUTPUT_PATH.name
        recommendations_csv = staging / RECOMMEND_OUTPUT_PATH.name
        report_path = staging / DIAGNOSTICS_OUTPUT_PATH.name
        eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")
        rec_df.to_csv(recommendations_csv, index=False, encoding="utf-8-sig")
        if diagnostics:
            write_diagnostics(
                report_path, test_df, y_true, metrics, train_count=train_count, global_mean=global_mean,
                zone_stats=stats["action_zone"], actual_sources=actual_sources, candidate_counts=candidate_counts,
                result_paths=(eval_csv, recommendations_csv),
            )

        # Invalidate the older report before exposing a new CSV.
        DIAGNOSTICS_OUTPUT_PATH.unlink(missing_ok=True)
        eval_csv.replace(EVAL_OUTPUT_PATH)
        recommendations_csv.replace(RECOMMEND_OUTPUT_PATH)
        if diagnostics:
            report_path.replace(DIAGNOSTICS_OUTPUT_PATH)
            print("진단 저장:", DIAGNOSTICS_OUTPUT_PATH)

    print("\n===== 실제 action delta_run_exp 예측 성능 =====")
    print(eval_df.to_string(index=False))
    print("저장:", EVAL_OUTPUT_PATH)

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
        help="Compare observed-action errors and candidate coverage against simple baselines.",
    )
    main(diagnostics=parser.parse_args().diagnostics)
