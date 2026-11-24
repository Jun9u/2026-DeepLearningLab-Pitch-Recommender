from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.run_paths import run_directory  # noqa: E402
from web.backend.frontend import add_frontend_routes  # noqa: E402
from web.backend.schemas import CompareRequest, RecommendRequest  # noqa: E402


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

@asynccontextmanager
async def lifespan(_app):
    paths = (LSTM_ARTIFACT_PATH, OUTCOME_ARTIFACT_PATH)
    missing_artifacts = [path.name for path in paths if not path.is_file()]
    service = None
    if not missing_artifacts:
        from web.backend.service import PitchRecommendationService

        service = PitchRecommendationService(*paths)
    yield {"service": service, "missing_artifacts": missing_artifacts}


def require_service(request: Request):
    service = request.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="서빙 모델 파일을 준비한 뒤 서버를 재시작하세요.")
    return service


app = FastAPI(
    title="Pitch Recommendation API",
    description="LSTM 기반 구종 후보 생성 + delta_run_exp 기반 zone 랭킹 추천 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_frontend_routes(app, FRONTEND_PATH)


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/health", responses={503: {"description": "서빙 모델 파일 준비 필요"}})
def health(request: Request):
    service = request.state.service
    if service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "missing_artifacts": request.state.missing_artifacts},
        )
    return {
        "status": "ok",
        "device": str(service.device),
        "num_pitchers_lstm_vocab": len(service.pitcher_vocab),
        "num_pitchers_outcome": len(service.pitcher_top_pitches),
        "valid_zones": service.valid_zones,
        "supported_pitch_types": service.supported_pitch_types,
    }


@app.get("/meta", responses={503: {"description": "서빙 모델 파일 준비 필요"}})
def meta(request: Request):
    service = require_service(request)
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


@app.post("/recommend", responses={503: {"description": "서빙 모델 파일 준비 필요"}})
def recommend(req: RecommendRequest, request: Request):
    return require_service(request).recommend(req)


@app.post("/compare", responses={503: {"description": "서빙 모델 파일 준비 필요"}})
def compare(req: CompareRequest, request: Request):
    return require_service(request).compare(req)
