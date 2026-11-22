"""Request models shared by the HTTP routes and recommendation service."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    # 선수 정보
    pitcher: int
    batter: Optional[int] = None

    # 현재 경기 상황
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    outs_when_up: int = Field(ge=0, le=2)
    inning: int = Field(ge=1, le=20)

    runner_1b: int = Field(default=0, ge=0, le=1)
    runner_2b: int = Field(default=0, ge=0, le=1)
    runner_3b: int = Field(default=0, ge=0, le=1)

    # 좌우 정보
    stand: str = Field(default="R")      # batter side: R/L
    p_throws: str = Field(default="R")   # pitcher hand: R/L

    # 같은 타석 내 이전 투구 정보
    # 5개보다 적어도 됨. 서버에서 왼쪽 PAD 처리.
    prev_pitch_types: List[str] = Field(default_factory=list)
    prev_zones: List[str] = Field(default_factory=list)

    # 사용자가 후보 구종을 직접 제한하고 싶을 때 사용
    # 예: ["FF", "SL", "CH"]
    candidate_pitch_types: Optional[List[str]] = None

    # 최종 추천 개수
    top_n: int = Field(default=5, ge=1, le=20)


class CompareRequest(RecommendRequest):
    action_pitch_type: str
    action_zone: str


def req_to_dict(req: RecommendRequest) -> Dict[str, Any]:
    """
    Pydantic v1/v2 호환용.
    """
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()
