"""Normalize and encode pitch inputs for serving."""

from decimal import Decimal, InvalidOperation

from src.sequence_helpers import HISTORY_LENGTH, left_pad


def normalize_zone(z, *, require_integral=False):
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
        if require_integral:
            exact_value = Decimal(z)
            if not exact_value.is_finite() or exact_value != exact_value.to_integral_value():
                return "ZUNK"
        return f"Z{int(float(z))}"
    except (InvalidOperation, ValueError, OverflowError):
        return "ZUNK"


def encode_sequence(tokens, vocab, max_len=HISTORY_LENGTH, pad_token="PAD", unk_token="UNK"):
    tokens = left_pad(tokens, max_len, pad_token)
    unk_id = vocab.get(unk_token, 1)
    return [vocab.get(token, unk_id) for token in tokens]


def encode_category(value, vocab, unk_token="UNK"):
    if value is None:
        return vocab.get(unk_token, 0)

    return vocab.get(str(value), vocab.get(unk_token, 0))


def clean_pitch_token(p, *, keep_other=False):
    if p is None:
        return None

    p = str(p).strip().upper()

    if p in ["", "NAN", "NONE"] or (p == "OTHER" and not keep_other):
        return None

    return p


def encode_pitch_history(prev_pitch_types, prev_zones, pitch_vocab, zone_vocab, *, other_pitches=()):
    """Keep one token per history entry so paired pitches and zones stay aligned."""
    pitches = [clean_pitch_token(pitch, keep_other=True) or "UNK" for pitch in prev_pitch_types]
    pitches = ["OTHER" if pitch in other_pitches else pitch for pitch in pitches]
    zones = [normalize_zone(zone) for zone in prev_zones]

    return (
        encode_sequence(pitches, pitch_vocab),
        encode_sequence(zones, zone_vocab, pad_token="ZPAD", unk_token="ZUNK"),
    )
