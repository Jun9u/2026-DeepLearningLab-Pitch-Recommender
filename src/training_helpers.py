"""Dataset, training and evaluation shared by pitch classifiers."""

import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset

if __package__:
    from . import training_features as features
else:
    import training_features as features


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def topk_accuracy_score(y_true, y_topk):
    correct = 0

    for true_label, pred_list in zip(y_true, y_topk):
        if true_label in pred_list:
            correct += 1

    return correct / len(y_true)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PitchSequenceDataset(Dataset):
    def __init__(
        self,
        df,
        pitch_vocab,
        zone_vocab,
        pitcher_vocab,
        stand_vocab,
        throws_vocab,
        target_to_idx,
        scaler,
        max_len=5,
    ):
        self.df = df.reset_index(drop=True)

        self.pitch_vocab = pitch_vocab
        self.zone_vocab = zone_vocab
        self.pitcher_vocab = pitcher_vocab
        self.stand_vocab = stand_vocab
        self.throws_vocab = throws_vocab
        self.target_to_idx = target_to_idx
        self.scaler = scaler
        self.max_len = max_len

        numeric_data = self.df[features.NUMERIC_COLS].astype(float).values
        self.numeric = self.scaler.transform(numeric_data).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq_pitch = features.encode_sequence(
            row[features.SEQ_PITCH_COL],
            self.pitch_vocab,
            max_len=self.max_len,
            pad_token="PAD",
            unk_token="UNK",
        )

        seq_zone = features.encode_sequence(
            row[features.SEQ_ZONE_COL],
            self.zone_vocab,
            max_len=self.max_len,
            pad_token="ZPAD",
            unk_token="ZUNK",
        )

        pitcher = features.encode_category(row[features.PITCHER_COL], self.pitcher_vocab)
        stand = features.encode_category(row[features.STAND_COL], self.stand_vocab)
        p_throws = features.encode_category(row[features.THROWS_COL], self.throws_vocab)

        label = self.target_to_idx[row[features.TARGET_COL]]

        return {
            "seq_pitch": torch.tensor(seq_pitch, dtype=torch.long),
            "seq_zone": torch.tensor(seq_zone, dtype=torch.long),
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "pitcher": torch.tensor(pitcher, dtype=torch.long),
            "stand": torch.tensor(stand, dtype=torch.long),
            "p_throws": torch.tensor(p_throws, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        seq_pitch = batch["seq_pitch"].to(device)
        seq_zone = batch["seq_zone"].to(device)
        numeric = batch["numeric"].to(device)
        pitcher = batch["pitcher"].to(device)
        stand = batch["stand"].to(device)
        p_throws = batch["p_throws"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            seq_pitch=seq_pitch,
            seq_zone=seq_zone,
            numeric=numeric,
            pitcher=pitcher,
            stand=stand,
            p_throws=p_throws,
        )

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def evaluate(model, loader, criterion, device, idx_to_target, return_predictions=False):
    model.eval()

    total_loss = 0.0
    total_count = 0

    y_true = []
    y_pred = []
    y_top3 = []

    with torch.no_grad():
        for batch in loader:
            seq_pitch = batch["seq_pitch"].to(device)
            seq_zone = batch["seq_zone"].to(device)
            numeric = batch["numeric"].to(device)
            pitcher = batch["pitcher"].to(device)
            stand = batch["stand"].to(device)
            p_throws = batch["p_throws"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                seq_pitch=seq_pitch,
                seq_zone=seq_zone,
                numeric=numeric,
                pitcher=pitcher,
                stand=stand,
                p_throws=p_throws,
            )

            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            pred = torch.argmax(logits, dim=1)

            k = min(3, logits.size(1))
            top3 = torch.topk(logits, k=k, dim=1).indices

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_top3.extend(top3.cpu().numpy().tolist())

    avg_loss = total_loss / total_count
    acc = accuracy_score(y_true, y_pred)
    top3_acc = topk_accuracy_score(y_true, y_top3)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    if not return_predictions:
        return metrics

    pred_names = [idx_to_target[i] for i in y_pred]
    true_names = [idx_to_target[i] for i in y_true]
    top3_names = [
        [idx_to_target[i] for i in pred_list]
        for pred_list in y_top3
    ]

    return metrics, true_names, pred_names, top3_names
