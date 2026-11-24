import time
import platform

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

if __package__:
    from . import training_features as features
    from . import training_helpers as training
    from .run_paths import run_directory
    from .game_split import split_indices
else:
    import training_features as features
    import training_helpers as training
    from run_paths import run_directory
    from game_split import split_indices


# =========================
# 설정
# =========================

EXPERIMENT_NAME = "lstm_weighted_macro"

RUN_DIR = run_directory()
DATA_PATH = RUN_DIR / "data/processed/sequence_data.csv"

MODEL_PATH = RUN_DIR / f"models/{EXPERIMENT_NAME}_best.pt"
RESULT_PATH = RUN_DIR / f"results/{EXPERIMENT_NAME}_result.csv"
PRED_PATH = RUN_DIR / f"results/{EXPERIMENT_NAME}_predictions.csv"
HISTORY_PATH = RUN_DIR / f"results/{EXPERIMENT_NAME}_training_log.csv"
REPORT_PATH = RUN_DIR / f"results/{EXPERIMENT_NAME}_classification_report.csv"

SEED = 42
N_EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

PITCH_EMB_DIM = 32
ZONE_EMB_DIM = 8
PITCHER_EMB_DIM = 16
STAND_EMB_DIM = 4
THROWS_EMB_DIM = 4

HIDDEN_DIM = 96
NUM_LAYERS = 1
DROPOUT = 0.2

USE_CLASS_WEIGHTS = True
BEST_METRIC = "macro_f1"  # "accuracy", "top3_accuracy", "macro_f1", "weighted_f1"


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================
# Model
# =========================

class LSTMPitchModel(nn.Module):
    def __init__(
        self,
        num_pitch_tokens,
        num_zone_tokens,
        num_pitchers,
        num_stands,
        num_throws,
        num_numeric_features,
        num_classes,
    ):
        super().__init__()

        self.pitch_emb = nn.Embedding(
            num_embeddings=num_pitch_tokens,
            embedding_dim=PITCH_EMB_DIM,
            padding_idx=0,
        )

        self.zone_emb = nn.Embedding(
            num_embeddings=num_zone_tokens,
            embedding_dim=ZONE_EMB_DIM,
            padding_idx=0,
        )

        lstm_input_dim = PITCH_EMB_DIM + ZONE_EMB_DIM

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0 if NUM_LAYERS == 1 else DROPOUT,
        )

        self.pitcher_emb = nn.Embedding(
            num_embeddings=num_pitchers,
            embedding_dim=PITCHER_EMB_DIM,
        )

        self.stand_emb = nn.Embedding(
            num_embeddings=num_stands,
            embedding_dim=STAND_EMB_DIM,
        )

        self.throws_emb = nn.Embedding(
            num_embeddings=num_throws,
            embedding_dim=THROWS_EMB_DIM,
        )

        final_input_dim = (
            HIDDEN_DIM
            + PITCHER_EMB_DIM
            + STAND_EMB_DIM
            + THROWS_EMB_DIM
            + num_numeric_features
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, num_classes),
        )

    def forward(self, seq_pitch, seq_zone, numeric, pitcher, stand, p_throws):
        pitch_emb = self.pitch_emb(seq_pitch)
        zone_emb = self.zone_emb(seq_zone)

        seq_x = torch.cat([pitch_emb, zone_emb], dim=-1)

        _, (hidden, _) = self.lstm(seq_x)

        seq_repr = hidden[-1]

        pitcher_repr = self.pitcher_emb(pitcher)
        stand_repr = self.stand_emb(stand)
        throws_repr = self.throws_emb(p_throws)

        x = torch.cat(
            [
                seq_repr,
                numeric,
                pitcher_repr,
                stand_repr,
                throws_repr,
            ],
            dim=1,
        )

        logits = self.classifier(x)

        return logits


# =========================
# main
# =========================

def main():
    total_start_time = time.perf_counter()

    training.set_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = [
        features.SEQ_PITCH_COL,
        features.SEQ_ZONE_COL,
        features.TARGET_COL,
        features.PITCHER_COL,
        features.STAND_COL,
        features.THROWS_COL,
    ] + features.NUMERIC_COLS

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    sort_cols = [
        c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    train, valid, test = split_indices(df)
    train_df = df.iloc[train].copy()
    valid_df = df.iloc[valid].copy()
    test_df = df.iloc[test].copy()

    print("===== 실행 환경 =====")
    print("experiment:", EXPERIMENT_NAME)
    print("python:", platform.python_version())
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())

    print("\n===== 데이터 분할 =====")
    print("전체:", df.shape)
    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    # target mapping은 train 기준
    target_classes = sorted(train_df[features.TARGET_COL].dropna().unique().tolist())
    target_to_idx = {label: idx for idx, label in enumerate(target_classes)}
    idx_to_target = {idx: label for label, idx in target_to_idx.items()}

    valid_df = features.filter_evaluation_targets(valid_df, target_classes, "Validation")
    test_df = features.filter_evaluation_targets(test_df, target_classes, "Test")

    print("\n===== 클래스 목록 =====")
    print(target_classes)
    print("클래스 수:", len(target_classes))

    print("\n===== Train target 분포 =====")
    print(train_df[features.TARGET_COL].value_counts())

    print("\n===== Valid target 분포 =====")
    print(valid_df[features.TARGET_COL].value_counts())

    print("\n===== Test target 분포 =====")
    print(test_df[features.TARGET_COL].value_counts())

    # vocab은 train 기준으로만 생성
    pitch_vocab = features.build_vocab_from_sequences(
        train_df[features.SEQ_PITCH_COL],
        special_tokens=["PAD", "UNK"],
    )

    zone_vocab = features.build_vocab_from_sequences(
        train_df[features.SEQ_ZONE_COL],
        special_tokens=["ZPAD", "ZUNK"],
    )

    pitcher_vocab = features.build_vocab_from_column(
        train_df[features.PITCHER_COL],
        special_tokens=["UNK"],
    )

    stand_vocab = features.build_vocab_from_column(
        train_df[features.STAND_COL],
        special_tokens=["UNK"],
    )

    throws_vocab = features.build_vocab_from_column(
        train_df[features.THROWS_COL],
        special_tokens=["UNK"],
    )

    print("\n===== Vocab 크기 =====")
    print("pitch_vocab  :", len(pitch_vocab))
    print("zone_vocab   :", len(zone_vocab))
    print("pitcher_vocab:", len(pitcher_vocab))
    print("stand_vocab  :", len(stand_vocab))
    print("throws_vocab :", len(throws_vocab))

    # numeric scaling
    scaler = StandardScaler()
    scaler.fit(train_df[features.NUMERIC_COLS].astype(float).values)

    train_dataset = training.PitchSequenceDataset(
        train_df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    )

    valid_dataset = training.PitchSequenceDataset(
        valid_df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    )

    test_dataset = training.PitchSequenceDataset(
        test_df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n사용 device:", device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = LSTMPitchModel(
        num_pitch_tokens=len(pitch_vocab),
        num_zone_tokens=len(zone_vocab),
        num_pitchers=len(pitcher_vocab),
        num_stands=len(stand_vocab),
        num_throws=len(throws_vocab),
        num_numeric_features=len(features.NUMERIC_COLS),
        num_classes=len(target_classes),
    ).to(device)

    print("\n===== 모델 정보 =====")
    print("trainable parameters:", training.count_parameters(model))

    # class imbalance 보정
    if USE_CLASS_WEIGHTS:
        train_label_idx = train_df[features.TARGET_COL].map(target_to_idx)
        counts = train_label_idx.value_counts().sort_index()

        weights = []

        for i in range(len(target_classes)):
            count = counts.get(i, 1)
            weights.append(1.0 / np.sqrt(count))

        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.mean()

        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        print("\n===== Class weights =====")
        for i, w in enumerate(weights):
            print(idx_to_target[i], round(float(w), 4))

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_score = -1.0
    best_epoch = 0
    history = []

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n===== 학습 시작 =====")

    train_start_time = time.perf_counter()

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start_time = time.perf_counter()

        train_loss = training.train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_metrics = training.evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            idx_to_target=idx_to_target,
        )

        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - train_start_time
        avg_epoch_time = elapsed_time / epoch
        eta_time = avg_epoch_time * (N_EPOCHS - epoch)

        score = valid_metrics[BEST_METRIC]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_top3_accuracy": valid_metrics["top3_accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "valid_weighted_f1": valid_metrics["weighted_f1"],
                "epoch_time_sec": round(epoch_time, 4),
                "elapsed_time_sec": round(elapsed_time, 4),
                "eta_time_sec": round(eta_time, 4),
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc={valid_metrics['accuracy']:.4f} | "
            f"valid_top3={valid_metrics['top3_accuracy']:.4f} | "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f} | "
            f"valid_weighted_f1={valid_metrics['weighted_f1']:.4f} | "
            f"epoch_time={training.format_time(epoch_time)} | "
            f"elapsed={training.format_time(elapsed_time)} | "
            f"eta={training.format_time(eta_time)}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_to_idx": target_to_idx,
                    "idx_to_target": idx_to_target,
                    "pitch_vocab": pitch_vocab,
                    "zone_vocab": zone_vocab,
                    "pitcher_vocab": pitcher_vocab,
                    "stand_vocab": stand_vocab,
                    "throws_vocab": throws_vocab,
                    "numeric_cols": features.NUMERIC_COLS,

                    "scaler_mean": scaler.mean_.astype(float).tolist(),
                    "scaler_scale": scaler.scale_.astype(float).tolist(),

                    "best_metric": BEST_METRIC,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "experiment_name": EXPERIMENT_NAME,
                    "use_class_weights": USE_CLASS_WEIGHTS,
                    "model_config": {
                        "pitch_emb_dim": PITCH_EMB_DIM,
                        "zone_emb_dim": ZONE_EMB_DIM,
                        "pitcher_emb_dim": PITCHER_EMB_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                    },
                },
                MODEL_PATH,
            )



    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

    print("\n===== Best model =====")
    print("best_epoch:", best_epoch)
    print(f"best_valid_{BEST_METRIC}:", best_score)
    print("모델 저장:", MODEL_PATH)
    print("학습 로그 저장:", HISTORY_PATH)

    # best model 로드 후 test 평가
    checkpoint = safe_torch_load(MODEL_PATH, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, true_names, pred_names, top3_names = training.evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        idx_to_target=idx_to_target,
        return_predictions=True,
    )

    total_time = time.perf_counter() - total_start_time

    print("\n===== Test 결과 =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n===== 실행 시간 =====")
    print("전체 실행 시간:", training.format_time(total_time))
    print("전체 실행 시간_sec:", round(total_time, 2))

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "experiment_name": EXPERIMENT_NAME,
                "best_epoch": best_epoch,
                "best_metric": BEST_METRIC,
                "best_valid_score": best_score,
                "use_class_weights": USE_CLASS_WEIGHTS,
                "device": str(device),
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "total_time_sec": round(total_time, 4),
                "trainable_parameters": training.count_parameters(model),
                **test_metrics,
            }
        ]
    )

    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["true_pitch_type"] = true_names
    pred_df["pred_pitch_type"] = pred_names
    pred_df["top3_pitch_type"] = ["|".join(x) for x in top3_names]

    pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

    report_dict = classification_report(
        true_names,
        pred_names,
        labels=target_classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(REPORT_PATH, encoding="utf-8-sig")

    print("\n결과 저장:", RESULT_PATH)
    print("예측 결과 저장:", PRED_PATH)
    print("분류 리포트 저장:", REPORT_PATH)


if __name__ == "__main__":
    main()
