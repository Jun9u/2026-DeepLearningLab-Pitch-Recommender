from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.run_paths import run_directory  # noqa: E402
from web.backend.schemas import CompareRequest, RecommendRequest  # noqa: E402
from web.backend.service import PitchRecommendationService  # noqa: E402


# ============================================================
# Artifact paths
# ============================================================

RUN_DIR = run_directory(PROJECT_ROOT)
LSTM_ARTIFACT_PATH = RUN_DIR / "models" / "lstm_serving_artifact.pt"
OUTCOME_ARTIFACT_PATH = RUN_DIR / "models" / "outcome_ranker_artifact.pkl"

FRONTEND_PATH = PROJECT_ROOT / "web" / "frontend" / "index.html"


# ============================================================
# FastAPI app
# ============================================================

service = PitchRecommendationService(LSTM_ARTIFACT_PATH, OUTCOME_ARTIFACT_PATH)

app = FastAPI(
    title="Pitch Recommendation API",
    description="LSTM 기반 구종 후보 생성 + delta_run_exp 기반 zone 랭킹 추천 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)

    return HTMLResponse(
        """
        <h1>Pitch Recommendation API</h1>
        <p>Frontend file not found. Use <a href="/docs">/docs</a> to test API.</p>
        """
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(service.device),
        "num_pitchers_lstm_vocab": len(service.pitcher_vocab),
        "num_pitchers_outcome": len(service.pitcher_top_pitches),
        "valid_zones": service.valid_zones,
        "supported_pitch_types": service.supported_pitch_types,
    }


@app.get("/meta")
def meta():
    return {
        "valid_zones": service.valid_zones,
        "supported_pitch_types": service.supported_pitch_types,
        "global_top_pitches": service.global_top_pitches,
        "supported_stand": list(service.stand_vocab.keys()),
        "supported_p_throws": list(service.throws_vocab.keys()),
        "request_example": {
            "pitcher": 622503,
            "batter": None,
            "balls": 1,
            "strikes": 2,
            "outs_when_up": 1,
            "inning": 7,
            "runner_1b": 0,
            "runner_2b": 1,
            "runner_3b": 0,
            "stand": "R",
            "p_throws": "R",
            "prev_pitch_types": ["FF", "SL", "FF", "CH", "SL"],
            "prev_zones": ["Z5", "Z13", "Z2", "Z14", "Z7"],
            "candidate_pitch_types": ["FF", "SL", "CH"],
            "top_n": 5,
        },
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return service.recommend(req)


@app.post("/compare")
def compare(req: CompareRequest):
    return service.compare(req)
