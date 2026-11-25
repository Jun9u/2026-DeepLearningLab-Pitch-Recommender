"""Request models shared by the HTTP routes and recommendation service."""

from typing import Annotated, Any, Dict, List, Optional, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError


PitchToken = Annotated[str, StringConstraints(max_length=32)]


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
    stand: str = Field(default="R", max_length=8)      # batter side: R/L
    p_throws: str = Field(default="R", max_length=8)   # pitcher hand: R/L

    # 같은 타석 내 이전 투구 정보
    # 5개보다 적어도 됨. 서버에서 왼쪽 PAD 처리.
    prev_pitch_types: List[PitchToken] = Field(default_factory=list, max_length=64)
    prev_zones: List[PitchToken] = Field(default_factory=list, max_length=64)

    # 사용자가 후보 구종을 직접 제한하고 싶을 때 사용
    # 예: ["FF", "SL", "CH"]
    candidate_pitch_types: Optional[List[PitchToken]] = Field(default=None, max_length=32)

    # 최종 추천 개수
    top_n: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_pitch_history(self) -> Self:
        if len(self.prev_pitch_types) != len(self.prev_zones):
            raise PydanticCustomError(
                "pitch_history_length_mismatch",
                "이전 투구의 구종과 위치는 같은 개수로 입력해 주세요.",
            )
        return self


class CompareRequest(RecommendRequest):
    action_pitch_type: PitchToken
    action_zone: PitchToken


def req_to_dict(req: RecommendRequest) -> Dict[str, Any]:
    """요청 모델을 응답에 포함할 딕셔너리로 변환합니다."""
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()
