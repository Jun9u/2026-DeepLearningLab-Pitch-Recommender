"""Feature columns and vocabulary encoding shared by pitch classifiers."""

import pandas as pd


TARGET_COL = "target_pitch_type"

SEQ_PITCH_COL = "seq_pitch_type"
SEQ_ZONE_COL = "seq_zone"

NUMERIC_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "runner_1b",
    "runner_2b",
    "runner_3b",
]

PITCHER_COL = "pitcher"
STAND_COL = "stand"
THROWS_COL = "p_throws"


def split_tokens(text):
    if pd.isna(text):
        return []
    return str(text).split("|")


def build_vocab_from_sequences(series, special_tokens):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for text in series.dropna():
        for token in split_tokens(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def build_vocab_from_column(series, special_tokens):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for value in series.dropna().astype(str):
        if value not in vocab:
            vocab[value] = len(vocab)

    return vocab


def encode_sequence(text, vocab, max_len=5, pad_token="PAD", unk_token="UNK"):
    tokens = split_tokens(text)

    if len(tokens) > max_len:
        tokens = tokens[-max_len:]

    if len(tokens) < max_len:
        tokens = [pad_token] * (max_len - len(tokens)) + tokens

    unk_id = vocab.get(unk_token, 1)

    return [vocab.get(token, unk_id) for token in tokens]


def encode_category(value, vocab, unk_token="UNK"):
    if pd.isna(value):
        return vocab[unk_token]

    return vocab.get(str(value), vocab[unk_token])
