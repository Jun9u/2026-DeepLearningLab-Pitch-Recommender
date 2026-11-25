"""Load serving artifacts and score pitch recommendations and comparisons."""

from pathlib import Path
import pickle

import numpy as np
import torch

from src.empirical_ranker import score_action, select_candidate_pitches
from src.train_lstm import load_serving_model
from src.training_features import encode_numeric_features
from .pitch_inputs import clean_pitch_token, encode_category, encode_pitch_history, normalize_zone
from .schemas import CompareRequest, RecommendRequest, req_to_dict


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class PitchRecommendationService:
    def __init__(self, lstm_artifact_path: Path, outcome_artifact_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not lstm_artifact_path.exists():
            raise FileNotFoundError(
                f"LSTM artifact가 없습니다: {lstm_artifact_path}\n"
                "먼저 python -u src/export_serving_artifacts.py 를 실행하세요."
            )

        if not outcome_artifact_path.exists():
            raise FileNotFoundError(
                f"Outcome artifact가 없습니다: {outcome_artifact_path}\n"
                "먼저 python -u src/export_serving_artifacts.py 를 실행하세요."
            )

        # -------------------------
        # Load LSTM artifact
        # -------------------------
        self.lstm_artifact = safe_torch_load(lstm_artifact_path, self.device)

        required_lstm_keys = [
            "model_state_dict",
            "target_to_idx",
            "idx_to_target",
            "pitch_vocab",
            "zone_vocab",
            "pitcher_vocab",
            "stand_vocab",
            "throws_vocab",
            "numeric_cols",
            "scaler_mean",
            "scaler_scale",
        ]

        missing = [k for k in required_lstm_keys if k not in self.lstm_artifact]
        if missing:
            raise ValueError(f"LSTM artifact 필수 키 누락: {missing}")

        self.pitch_vocab = self.lstm_artifact["pitch_vocab"]
        self.zone_vocab = self.lstm_artifact["zone_vocab"]
        self.pitcher_vocab = self.lstm_artifact["pitcher_vocab"]
        self.stand_vocab = self.lstm_artifact["stand_vocab"]
        self.throws_vocab = self.lstm_artifact["throws_vocab"]
        self.history_other_pitches = frozenset(self.lstm_artifact.get("history_other_pitches", ()))

        self.target_to_idx = self.lstm_artifact["target_to_idx"]
        self.idx_to_target = self.lstm_artifact["idx_to_target"]

        # torch 저장 dict에서 key가 str로 바뀌는 경우 방어
        self.idx_to_target = {
            int(k): v for k, v in self.idx_to_target.items()
        }

        self.numeric_cols = self.lstm_artifact["numeric_cols"]
        self.scaler_mean = np.array(self.lstm_artifact["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.array(self.lstm_artifact["scaler_scale"], dtype=np.float32)

        self.scaler_scale[self.scaler_scale == 0] = 1.0

        self.model = load_serving_model(self.lstm_artifact, self.device)

        # -------------------------
        # Load outcome artifact
        # -------------------------
        with open(outcome_artifact_path, "rb") as f:
            self.outcome_artifact = pickle.load(f)

        required_outcome_keys = [
            "global_mean",
            "alpha",
            "thresholds",
            "valid_zones",
            "pitcher_top_pitches",
            "global_top_pitches",
            "stats",
        ]

        missing = [k for k in required_outcome_keys if k not in self.outcome_artifact]
        if missing:
            raise ValueError(f"Outcome artifact 필수 키 누락: {missing}")

        self.valid_zones = self.outcome_artifact["valid_zones"]
        self.global_top_pitches = self.outcome_artifact["global_top_pitches"]
        self.pitcher_top_pitches = self.outcome_artifact["pitcher_top_pitches"]
        self.stats = self.outcome_artifact["stats"]

        # outcome artifact에 존재하는 실제 추천 가능 구종 목록
        self.supported_pitch_types = sorted(
            [
                key[0]
                for key in self.stats["action"].keys()
                if key and key[0] != "OTHER"
            ]
        )

        print("===== PitchRecommendationService loaded =====")
        print("device:", self.device)
        print("num_pitchers:", len(self.pitcher_vocab))
        print("valid_zones:", self.valid_zones)
        print("supported_pitch_types:", self.supported_pitch_types)

    # --------------------------------------------------------
    # LSTM Top-K pitch type prediction
    # --------------------------------------------------------
    def predict_lstm_topk(self, req: RecommendRequest, k=3):
        seq_pitch, seq_zone = encode_pitch_history(
            req.prev_pitch_types, req.prev_zones, self.pitch_vocab, self.zone_vocab,
            other_pitches=self.history_other_pitches,
        )

        numeric_values = encode_numeric_features(req, self.scaler_mean, self.scaler_scale)

        pitcher_id = encode_category(req.pitcher, self.pitcher_vocab)
        stand_id = encode_category(req.stand.upper(), self.stand_vocab)
        throws_id = encode_category(req.p_throws.upper(), self.throws_vocab)

        batch = {
            "seq_pitch": torch.tensor([seq_pitch], dtype=torch.long, device=self.device),
            "seq_zone": torch.tensor([seq_zone], dtype=torch.long, device=self.device),
            "numeric": torch.tensor(
                numeric_values.reshape(1, -1),
                dtype=torch.float32,
                device=self.device,
            ),
            "pitcher": torch.tensor([pitcher_id], dtype=torch.long, device=self.device),
            "stand": torch.tensor([stand_id], dtype=torch.long, device=self.device),
            "p_throws": torch.tensor([throws_id], dtype=torch.long, device=self.device),
        }

        with torch.no_grad():
            logits = self.model(**batch)
            probs = torch.softmax(logits, dim=1)[0]

            topk = torch.topk(
                probs,
                k=min(k, probs.shape[0]),
                dim=0,
            )

        results = []

        for idx, prob in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
            pitch_type = self.idx_to_target[int(idx)]

            results.append(
                {
                    "pitch_type": pitch_type,
                    "probability": float(prob),
                }
            )

        return results

    # --------------------------------------------------------
    # Outcome ranker
    # --------------------------------------------------------
    def _predict_delta(self, req: RecommendRequest, action_pitch_type: str, action_zone: str):
        return score_action(
            int(req.pitcher), int(req.balls), int(req.strikes),
            str(action_pitch_type).upper(), str(action_zone).upper(),
            stats=self.stats, thresholds=self.outcome_artifact["thresholds"],
            global_mean=float(self.outcome_artifact["global_mean"]),
            alpha=float(self.outcome_artifact["alpha"]), global_count=0,
        )

    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------
    def build_candidate_pitches(self, req: RecommendRequest, lstm_top3):
        candidate_pitches = []

        # 0. 사용자가 후보 구종을 직접 넣은 경우: 강한 제약으로 사용
        if req.candidate_pitch_types:
            for p in req.candidate_pitch_types:
                p = clean_pitch_token(p)

                if p is None:
                    continue

                if p not in self.supported_pitch_types:
                    continue

                if p not in candidate_pitches:
                    candidate_pitches.append(p)

            # 사용자가 직접 후보를 넣은 경우에는 pitcher fallback을 추가하지 않음
            if candidate_pitches:
                return candidate_pitches

        return select_candidate_pitches(
            (item["pitch_type"] for item in lstm_top3),
            self.pitcher_top_pitches.get(int(req.pitcher), []),
            self.global_top_pitches,
            self.supported_pitch_types,
        )

    def recommend(self, req: RecommendRequest):
        lstm_top3 = self.predict_lstm_topk(req, k=3)

        candidate_pitches = self.build_candidate_pitches(req, lstm_top3)

        candidates = []

        for pitch in candidate_pitches:
            for zone in self.valid_zones:
                score, source, count = self._predict_delta(req, pitch, zone)

                candidates.append(
                    {
                        "pitch_type": pitch,
                        "zone": zone,
                        "pred_delta_run_exp": float(score),
                        "source": source,
                        "source_count": int(count),
                    }
                )

        candidates = sorted(candidates, key=lambda x: x["pred_delta_run_exp"])
        recommendations = candidates[: req.top_n]

        warning_messages = []

        if len(req.prev_pitch_types) == 0:
            warning_messages.append(
                "이전 투구 정보가 없습니다. 현재 모델은 이전 투구 이력이 있는 상황에서 더 안정적으로 동작합니다."
            )

        return {
            "input": req_to_dict(req),
            "warnings": warning_messages,
            "lstm_top3": lstm_top3,
            "candidate_pitches": candidate_pitches,
            "recommendations": recommendations,
            "notes": {
                "zone_meaning": "추천 zone은 Statcast 실제 도착 zone 기반입니다. 포수의 의도 위치와는 다를 수 있습니다.",
                "delta_run_exp_meaning": "pred_delta_run_exp가 낮을수록 투수 관점에서 유리한 과거 기대득점 변화 경향을 의미합니다.",
            },
        }

    def compare(self, req: CompareRequest):
        pitch = clean_pitch_token(req.action_pitch_type)
        zone = normalize_zone(req.action_zone, require_integral=True)

        if pitch is None:
            return {
                "error": "유효하지 않은 구종입니다.",
                "input_pitch_type": req.action_pitch_type,
            }

        if pitch not in self.supported_pitch_types:
            return {
                "error": "지원하지 않는 구종입니다.",
                "input_pitch_type": pitch,
                "supported_pitch_types": self.supported_pitch_types,
            }

        if zone not in self.valid_zones:
            return {
                "error": "지원하지 않는 zone입니다.",
                "input_zone": zone,
                "valid_zones": self.valid_zones,
            }

        score, source, count = self._predict_delta(req, pitch, zone)

        # 같은 상황에서 추천 결과도 같이 계산
        rec_result = self.recommend(req)
        best = rec_result["recommendations"][0] if rec_result["recommendations"] else None

        result = {
            "input": req_to_dict(req),
            "selected_action": {
                "pitch_type": pitch,
                "zone": zone,
                "pred_delta_run_exp": float(score),
                "source": source,
                "source_count": int(count),
            },
            "best_recommendation": best,
        }

        if best is not None:
            result["comparison"] = {
                "selected_minus_best": float(score - best["pred_delta_run_exp"]),
                "is_selected_better_or_equal": bool(score <= best["pred_delta_run_exp"]),
            }

        return result
