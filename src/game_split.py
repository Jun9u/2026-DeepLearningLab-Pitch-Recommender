"""Build and apply one chronological game split across experiment entry points."""

import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

if __package__:
    from .raw_statcast import MODEL_CSV, raw_statcast_directory
    from .run_paths import run_directory
else:
    from raw_statcast import MODEL_CSV, raw_statcast_directory
    from run_paths import run_directory


GAME_COLUMNS = ["game_date", "game_pk"]
SPLITS = ["train", "validation", "test"]
GAME_SPLIT_HASH_COL = "game_split_sha256"


def game_keys(frame):
    missing = set(GAME_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Game split requires columns: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(frame["game_date"], format="%Y-%m-%d", errors="raise")
    ids = pd.to_numeric(frame["game_pk"], errors="raise")
    if dates.isna().any() or ids.isna().any() or not ((ids > 0) & (ids % 1 == 0)).all():
        raise ValueError("Game split requires a date and a positive integer game_pk on every row")
    return pd.DataFrame({"game_date": dates, "game_pk": ids.astype("int64")})


def build_game_split(frame):
    """Assign whole dates in order, targeting a 70/10/20 split by game count."""
    games = game_keys(frame).drop_duplicates()
    if games["game_pk"].duplicated().any():
        raise ValueError("A game_pk cannot have multiple dates in the game split")
    games = games.sort_values(GAME_COLUMNS).reset_index(drop=True)

    cumulative = games.groupby("game_date").size().cumsum()
    if len(cumulative) < 3:
        raise ValueError("Game split requires games on at least three dates")

    # Leave at least one date in each partition. A tie picks the earlier date.
    train_date = (cumulative.iloc[:-2] - len(games) * 0.7).abs().idxmin()
    eligible = cumulative.iloc[:-1]
    valid_date = (eligible[eligible.index > train_date] - len(games) * 0.8).abs().idxmin()

    games["split"] = "test"
    games.loc[games["game_date"] <= valid_date, "split"] = "validation"
    games.loc[games["game_date"] <= train_date, "split"] = "train"
    return games


def read_split_manifest():
    configured = os.environ.get("PITCH_GAME_SPLIT")
    return Path(configured).expanduser().read_bytes() if configured else None


def split_manifest_hash(manifest):
    return sha256(manifest).hexdigest() if manifest is not None else None


def split_indices(frame, *, with_validation=True, manifest_bytes=None):
    """Return positional train, validation and test selectors for a sorted frame."""
    if manifest_bytes is None:
        manifest_bytes = read_split_manifest()
    if manifest_bytes is None:
        test_start = int(len(frame) * 0.8)
        train_end = int(len(frame) * 0.7) if with_validation else test_start
        return slice(0, train_end), slice(train_end, test_start), slice(test_start, None)

    manifest = pd.read_csv(BytesIO(manifest_bytes), usecols=[*GAME_COLUMNS, "split"])
    manifest = game_keys(manifest).assign(split=manifest["split"])
    if manifest["game_pk"].duplicated().any() or set(manifest["split"]) != set(SPLITS):
        raise ValueError("Game split must assign each game once and include train, validation and test")
    dates = manifest.groupby("split")["game_date"].agg(["min", "max"]).loc[SPLITS]
    if not (dates["max"].iloc[:-1].to_numpy() < dates["min"].iloc[1:].to_numpy()).all():
        raise ValueError("Game split dates must be strictly ordered by partition")

    manifest = manifest.set_index("game_pk")
    keys = game_keys(frame)
    labels = keys["game_pk"].map(manifest["split"])
    if labels.isna().any() or not keys["game_date"].eq(keys["game_pk"].map(manifest["game_date"])).all():
        raise ValueError("Input game IDs and dates must match the game split manifest")

    train = labels.isin(["train"] if with_validation else ["train", "validation"])
    valid = labels.eq("validation") if with_validation else labels.eq("")
    test = labels.eq("test")
    if not train.any() or not test.any() or (with_validation and not valid.any()):
        raise ValueError("Input contains no rows for one or more required game split partitions")
    return train.to_numpy(), valid.to_numpy(), test.to_numpy()


def main():
    root = run_directory()
    source = raw_statcast_directory(root / "data/raw") / MODEL_CSV
    output = root / "data/processed/game_split.csv"
    manifest = build_game_split(pd.read_csv(source, usecols=GAME_COLUMNS))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, mode="x")
    print(manifest.groupby("split")["game_date"].agg(["size", "min", "max"]).loc[SPLITS].to_string())
    print(output)


if __name__ == "__main__":
    main()
