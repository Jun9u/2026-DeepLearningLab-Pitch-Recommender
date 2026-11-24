import pandas as pd

from src.training_features import (
    build_vocab_from_column,
    build_vocab_from_sequences,
    encode_category,
    encode_sequence,
)


def test_encode_sequence_preserves_first_seen_ids_and_recent_history():
    history = pd.Series([None, "ZPAD|Z9|Z2", "Z2|Z4"])
    vocab = build_vocab_from_sequences(history, ["ZPAD", "ZUNK"])

    assert vocab == {"ZPAD": 0, "ZUNK": 1, "Z9": 2, "Z2": 3, "Z4": 4}
    options = {"max_len": 3, "pad_token": "ZPAD", "unk_token": "ZUNK"}
    assert encode_sequence("Z4|Z9|Z2|Z11", vocab, **options) == [2, 3, 1]
    assert encode_sequence("Z4", vocab, **options) == [0, 0, 4]
    assert encode_sequence(None, vocab, **options) == [0, 0, 0]


def test_encode_category_stringifies_ids_and_falls_back_to_unknown():
    pitchers = pd.Series([900002, pd.NA, 900001, 900002], dtype="Int64")
    vocab = build_vocab_from_column(pitchers, ["UNK"])

    assert vocab == {"UNK": 0, "900002": 1, "900001": 2}
    assert encode_category(900001, vocab) == 2
    assert encode_category(900003, vocab) == 0
    assert encode_category(pd.NA, vocab) == 0
