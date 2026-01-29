from fastapi import APIRouter, Query
from typing import Optional
from database import db
from schemas.schemas import HealthResponse

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health", response_model=HealthResponse)
async def get_health_status(
    session_id: Optional[str] = Query(None, description="세션 ID ('latest', 숫자, 또는 생략)")
):
    """
    시스템 건강 상태 모니터링 API

    - session_id="latest": 진행 중인 세션
    - session_id=숫자: 특정 세션
    - session_id 생략: 전체 데이터

    **알림 임계값:**
    - 불량률: warning 10%, critical 20%
    - 평균 신뢰도: warning 0.85, critical 0.75
    - 저신뢰도 비율: warning 20%, critical 40%
    - PCB당 평균 결함: warning 2.0개, critical 3.0개
    """
    return await db.get_health(session_id)
