import unittest

import numpy as np

from src.training_features import encode_numeric_features
from web.backend.pitch_inputs import (
    clean_pitch_token,
    encode_category,
    encode_pitch_history,
    encode_sequence,
    normalize_zone,
)
from web.backend.schemas import RecommendRequest


class PitchInputTests(unittest.TestCase):
    def test_normalize_zone_accepts_labels_and_numeric_input(self):
        self.assertEqual(normalize_zone(" z13 "), "Z13")
        self.assertEqual(normalize_zone("2.0"), "Z2")
        self.assertEqual(normalize_zone(None), "ZUNK")
        self.assertEqual(normalize_zone(" NAn "), "ZUNK")
        self.assertEqual(normalize_zone("unknown"), "ZUNK")

    def test_clean_pitch_token_excludes_other_and_missing_values(self):
        self.assertEqual(clean_pitch_token(" ff "), "FF")
        self.assertIsNone(clean_pitch_token(None))
        self.assertIsNone(clean_pitch_token(" other "))

    def test_normalize_zone_treats_nonfinite_values_as_unknown(self):
        for zone in ("infinity", "-Inf", "1e400", float("inf"), float("-inf")):
            with self.subTest(zone=zone):
                self.assertEqual(normalize_zone(zone), "ZUNK")

    def test_fractional_zones_cannot_be_scored_as_integer_zones(self):
        self.assertEqual(normalize_zone("1.0", require_integral=True), "Z1")
        self.assertEqual(normalize_zone("1.9", require_integral=True), "ZUNK")
        self.assertEqual(normalize_zone("1.00000000000000000001", require_integral=True), "ZUNK")
        self.assertEqual(normalize_zone("1.9"), "Z1")

    def test_serving_encoders_keep_padding_and_unknown_ids(self):
        vocab = {"ZPAD": 0, "ZUNK": 1, "Z1": 2, "Z2": 3}
        options = {"max_len": 3, "pad_token": "ZPAD", "unk_token": "ZUNK"}

        self.assertEqual(encode_sequence(["Z1", "invalid"], vocab, **options), [0, 2, 1])
        self.assertEqual(encode_sequence(["Z1", "Z2", "invalid", "Z2"], vocab, **options), [3, 1, 3])
        self.assertEqual(encode_category(123, {"123": 7, "UNK": 1}), 7)
        self.assertEqual(encode_category(456, {"123": 7, "UNK": 1}), 1)
        self.assertEqual(encode_category(None, {"123": 7}), 0)

    def test_pitch_history_preserves_other_and_missing_slots(self):
        pitch_vocab = {"PAD": 0, "UNK": 1, "FF": 2, "SL": 3, "OTHER": 4}
        zone_vocab = {"ZPAD": 0, "ZUNK": 1, **{f"Z{i}": i + 1 for i in range(1, 7)}}
        zones = [f"Z{i}" for i in range(1, 7)]
        pitches = ["FF", " other ", "", "SL", "nan", "none"]

        seq_pitch, seq_zone = encode_pitch_history(pitches, zones, pitch_vocab, zone_vocab)
        self.assertEqual(list(zip(seq_pitch, seq_zone)), [(4, 3), (1, 4), (3, 5), (1, 6), (1, 7)])

        del pitch_vocab["OTHER"]
        seq_pitch, seq_zone = encode_pitch_history(["OTHER", "SL"], ["Z1", "Z2"], pitch_vocab, zone_vocab)
        self.assertEqual(list(zip(seq_pitch, seq_zone)), [(0, 0), (0, 0), (0, 0), (1, 2), (3, 3)])

    def test_artifact_rare_pitch_mapping_applies_to_history(self):
        pitch_vocab = {"PAD": 0, "UNK": 1, "OTHER": 2}
        zone_vocab = {"ZPAD": 0, "ZUNK": 1, "Z1": 2}
        codes, _ = encode_pitch_history(
            [" ep ", "X"], ["Z1", "Z1"], pitch_vocab, zone_vocab, other_pitches={"EP"},
        )
        self.assertEqual(codes, [0, 0, 0, 2, 1])

    def test_serving_numeric_features_apply_training_order_and_float32_scaler(self):
        mean = np.array([1, 0, 2, 3, 1, 0, 1], dtype=np.float32)
        scale = np.array([2, 2, 1, 4, 2, 4, 8], dtype=np.float32)
        scenarios = [
            ({"runner_1b": 1}, [1, 1, -1, 1, 0, 0, -0.125]),
            ({"runner_2b": 1}, [1, 1, -1, 1, -0.5, 0.25, -0.125]),
            ({"runner_3b": 1}, [1, 1, -1, 1, -0.5, 0, 0]),
        ]
        for runners, expected in scenarios:
            with self.subTest(runners=runners):
                request = RecommendRequest(pitcher=123, balls=3, strikes=2, outs_when_up=1, inning=7, **runners)
                numeric = encode_numeric_features(request, mean, scale)
                self.assertEqual(numeric.dtype, np.float32)
                np.testing.assert_array_equal(numeric, expected)
