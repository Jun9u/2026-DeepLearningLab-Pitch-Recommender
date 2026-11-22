"""Normalize and encode pitch inputs for serving."""

from src.sequence_helpers import left_pad


def normalize_zone(z):
    """
    입력 zone을 모델에서 쓰는 Z1, Z2 ... 형태로 통일.
    """
    if z is None:
        return "ZUNK"

    z = str(z).strip().upper()

    if z in ["", "NAN", "NONE"]:
        return "ZUNK"

    if z.startswith("Z"):
        return z

    try:
        return f"Z{int(float(z))}"
    except ValueError:
        return "ZUNK"


def encode_sequence(tokens, vocab, max_len=5, pad_token="PAD", unk_token="UNK"):
    tokens = left_pad(tokens, max_len, pad_token)
    unk_id = vocab.get(unk_token, 1)
    return [vocab.get(token, unk_id) for token in tokens]


def encode_category(value, vocab, unk_token="UNK"):
    if value is None:
        return vocab.get(unk_token, 0)

    return vocab.get(str(value), vocab.get(unk_token, 0))


def clean_pitch_token(p):
    if p is None:
        return None

    p = str(p).strip().upper()

    if p in ["", "NAN", "NONE", "OTHER"]:
        return None

    return p
