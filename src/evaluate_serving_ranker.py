"""Evaluate fixed serving rankers on observed pitches after the release's fit period."""

import argparse
from datetime import date
from hashlib import file_digest
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .empirical_ranker import COUNT_ZONE_MODE, EMPIRICAL_MODE, TARGET_COL, scoring_function
from .outcome_empirical_recommender import regression_metrics
from .ranker_artifact import RankerArtifactUnpickler
from .sequence_helpers import PITCH_KEY, PITCH_ORDER
from .serving_paths import serving_artifact_paths


CONTEXT_COLUMNS = ["pitcher", "balls", "strikes", "action_pitch_type_original", "action_zone"]


def evaluate_release(sequence_csv: Path, release_dir: Path):
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    ranker_manifest = manifest["ranker"]
    partitions = manifest["split"]["partitions"]
    fit_last_date = max(
        date.fromisoformat(partitions[name]["last_game_date"]) for name in ranker_manifest["fit_partitions"]
    )

    with sequence_csv.open("rb") as source:
        sequence_hash = file_digest(source, "sha256").hexdigest()
        source.seek(0)
        frame = pd.read_csv(source, usecols=[*PITCH_ORDER, *CONTEXT_COLUMNS, TARGET_COL])
    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
    if frame.empty or frame.isna().any().any() or not np.isfinite(frame[TARGET_COL]).all():
        raise ValueError("평가 CSV에는 관측값과 입력이 모두 유효한 투구가 있어야 합니다.")
    if frame.duplicated(PITCH_KEY).any():
        raise ValueError("평가 CSV에 중복 투구 키가 있습니다.")
    dates = pd.to_datetime(frame["game_date"], format="%Y-%m-%d").dt.date
    if dates.isna().any():
        raise ValueError("평가 CSV에 날짜가 없는 투구가 있습니다.")
    if dates.min() <= fit_last_date:
        raise ValueError(f"평가 투구 날짜는 랭커 적합 종료일 {fit_last_date} 이후여야 합니다.")
    frame = frame.sort_values(PITCH_ORDER)

    _, ranker_path = serving_artifact_paths(release_dir / "models")
    relative_path = ranker_path.relative_to(release_dir).as_posix()
    expected_hash = manifest["artifacts"][relative_path]["sha256"]
    with ranker_path.open("rb") as source:
        ranker_hash = file_digest(source, "sha256").hexdigest()
        if ranker_hash != expected_hash:
            raise ValueError(f"랭커 파일 SHA-256이 매니페스트와 일치하지 않습니다: {ranker_path}")
        source.seek(0)
        artifact = RankerArtifactUnpickler(source).load()

    stats = artifact["stats"]
    known_pitches = {pitch for (pitch,) in stats["action"]}
    original_pitches = frame["action_pitch_type_original"]
    mapped = original_pitches.where(original_pitches.isin(known_pitches), "OTHER")
    settings = {
        "stats": stats,
        "global_mean": artifact["global_mean"],
        "global_count": ranker_manifest["fit_rows"],
        "thresholds": artifact["thresholds"],
        "alpha": artifact["alpha"],
    }
    pair_stats = stats["action_zone"]
    unseen_pair = {"mean": artifact["global_mean"]}
    target = frame[TARGET_COL].to_numpy()
    contexts = list(
        frame[CONTEXT_COLUMNS].assign(action_pitch_type_original=mapped).itertuples(index=False, name=None)
    )

    forecasts = {}
    for mode in (EMPIRICAL_MODE, COUNT_ZONE_MODE):
        score = scoring_function(mode)
        forecasts[mode] = np.fromiter((score(*row, **settings)[0] for row in contexts), float, count=len(frame))
    forecasts["pitch_zone_mean"] = np.fromiter(
        (pair_stats.get(row[-2:], unseen_pair)["mean"] for row in contexts), float, count=len(frame),
    )

    metrics = {}
    for mode, prediction in forecasts.items():
        result = regression_metrics(target, prediction)
        if math.isnan(result["r2"]):
            result["r2"] = None
        metrics[mode] = result
    return {
        "ranker_sha256": ranker_hash,
        "sequence_sha256": sequence_hash,
        "fit_last_date": fit_last_date.isoformat(),
        "first_game_date": dates.min().isoformat(),
        "last_game_date": dates.max().isoformat(),
        "games": frame["game_pk"].nunique(),
        "pitches": len(frame),
        "mapped_to_other": int(mapped.ne(original_pitches).sum()),
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-csv", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True, help="Unpacked serving release with manifest.json.")
    args = parser.parse_args(argv)
    try:
        report = evaluate_release(args.sequence_csv, args.release_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
