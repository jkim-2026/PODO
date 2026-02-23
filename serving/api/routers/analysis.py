"""
VLM 기반 결함 분석 API 라우터

GET /analysis/{log_id} - 결함 상세 분석 (온디맨드, DB 캐싱)
"""
import json
import logging

from fastapi import APIRouter, HTTPException

from database import db
from utils import vlm_analyzer
from config.settings import GEMINI_API_KEY

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["Analysis"])


@router.get("/analysis/{log_id}")
async def get_analysis(log_id: int):
    """
    VLM(Gemini) 기반 결함 상세 분석

    - 캐시된 결과가 있으면 즉시 반환 (status: "cached")
    - 없으면 S3 이미지 다운로드 → 어노테이션/크롭 → Gemini API → DB 캐싱 (status: "analyzed")
    """
    # 1. 검사 로그 조회
    log = await db.get_inspection_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"검사 로그 {log_id}를 찾을 수 없습니다")

    # 2. 캐시 확인
    if log.get("vlm_analysis"):
        try:
            cached = json.loads(log["vlm_analysis"])
            return {
                "log_id": log_id,
                "status": "cached",
                "analysis": cached
            }
        except json.JSONDecodeError:
            # 캐시 데이터가 손상된 경우 재분석
            logger.warning(f"[VLM] 캐시 데이터 손상 (log_id={log_id}), 재분석 수행")

    # 3. detections 파싱
    detections = []
    if log.get("detections"):
        try:
            detections = json.loads(log["detections"])
        except json.JSONDecodeError:
            pass

    if not detections:
        raise HTTPException(status_code=400, detail="정상 PCB는 VLM 분석을 지원하지 않습니다")

    # 4. 이미지 경로 확인
    image_path = log.get("image_path")
    if not image_path:
        raise HTTPException(status_code=400, detail="이미지가 없어 분석할 수 없습니다")

    # 5. API 키 확인
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY가 설정되지 않았습니다")

    # 6. VLM 분석 수행
    try:
        analysis = await vlm_analyzer.analyze_defects(image_path, detections)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"[VLM] 분석 실패 (log_id={log_id}): {e}")
        raise HTTPException(status_code=503, detail=f"VLM 분석 실패: {e}")

    # 7. DB 캐싱
    try:
        await db.save_vlm_analysis(log_id, analysis)
        logger.info(f"[VLM] 분석 결과 캐싱 완료 (log_id={log_id})")
    except Exception as e:
        logger.error(f"[VLM] 캐싱 실패 (log_id={log_id}): {e}")
        # 캐싱 실패해도 결과는 반환

    return {
        "log_id": log_id,
        "status": "analyzed",
        "analysis": analysis
    }
