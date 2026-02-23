"""
VLM(Gemini) 기반 PCB 결함 분석 유틸리티

이미지 어노테이션/크롭 + Gemini API 호출로 결함 상세 분석 수행
"""
import base64
import io
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont

from config.settings import GEMINI_API_KEY, S3_BUCKET_NAME
from utils.image_utils import s3_client

logger = logging.getLogger("uvicorn")

# 한국 표준시
KST = timezone(timedelta(hours=9))

# 결함 유형별 색상 매핑 (RGB)
DEFECT_COLORS = {
    "missing_hole": (230, 25, 75),      # 빨강
    "mouse_bite": (60, 180, 75),        # 초록
    "open_circuit": (255, 225, 25),     # 노랑
    "short": (67, 99, 216),             # 파랑
    "spur": (245, 130, 49),             # 주황
    "spurious_copper": (145, 30, 180),  # 보라
}

# 정규화: 공백/대문자 → 소문자+언더스코어
def _normalize_defect_type(defect_type: str) -> str:
    return defect_type.strip().lower().replace(" ", "_")


def _get_color(defect_type: str) -> tuple:
    """결함 유형에 맞는 색상 반환"""
    normalized = _normalize_defect_type(defect_type)
    return DEFECT_COLORS.get(normalized, (255, 0, 0))


def _create_annotated_image(image_bytes: bytes, detections: List[Dict]) -> bytes:
    """
    원본 이미지에 bbox + 라벨을 그린 어노테이션 이미지 생성

    Args:
        image_bytes: 원본 이미지 바이트
        detections: 결함 목록 [{"defect_type", "confidence", "bbox"}]

    Returns:
        어노테이션된 이미지 JPEG 바이트
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 기본 폰트 사용
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        bbox = det.get("bbox", [0, 0, 0, 0])
        defect_type = det.get("defect_type", "unknown")
        confidence = det.get("confidence", 0)
        color = _get_color(defect_type)

        x1, y1, x2, y2 = bbox
        # bbox 그리기 (두께 3)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color
            )

        # 라벨 텍스트
        label = f"#{i} {defect_type} {confidence:.0%}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 라벨 배경
        label_y = max(y1 - text_h - 6, 0)
        draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 4], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    # JPEG으로 변환
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _crop_defect_image(image_bytes: bytes, bbox: List[int], padding: int = 20) -> bytes:
    """
    bbox 영역을 패딩 포함하여 크롭

    Args:
        image_bytes: 원본 이미지 바이트
        bbox: [x1, y1, x2, y2]
        padding: 크롭 영역 패딩 (픽셀)

    Returns:
        크롭된 이미지 JPEG 바이트
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    x1, y1, x2, y2 = bbox
    # 패딩 적용 + 범위 클램핑
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(w, x2 + padding)
    crop_y2 = min(h, y2 + padding)

    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_prompt(detections: List[Dict]) -> str:
    """
    Gemini API용 분석 프롬프트 생성

    Args:
        detections: 결함 목록

    Returns:
        프롬프트 문자열
    """
    # 결함 정보 블록 구성
    defect_blocks = []
    for i, det in enumerate(detections):
        defect_type = det.get("defect_type", "unknown")
        confidence = det.get("confidence", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        # 크롭 이미지 번호: 어노테이션 이미지가 1번째이므로, 크롭은 2번째부터
        crop_num = i + 2
        defect_blocks.append(
            f"결함 #{i}: 유형={defect_type}, 신뢰도={confidence:.2f}, 위치={bbox} (크롭 이미지: {crop_num}번째)"
        )

    defects_text = "\n".join(defect_blocks)

    prompt = f"""당신은 IPC-A-600 표준 기반 PCB 품질 검사 전문 엔지니어입니다.

## PCB 결함 유형 정의

1. **Missing Hole** (홀 누락): PCB 드릴링 공정에서 홀이 형성되지 않은 결함. 부품 삽입 불가로 이어짐. 전기적 연결 불량 위험.
2. **Mouse Bite** (마우스 바이트): PCB 분리 공정에서 발생하는 톱니 형태의 불규칙한 가장자리. V-scoring 공정 불량 관련. spur와 구별: mouse_bite는 보드 가장자리에 위치.
3. **Open Circuit** (개방 회로): 도체 패턴이 끊어진 결함. 에칭 과다 또는 기계적 손상이 원인. 회로 단절로 기능 불량 직결.
4. **Short** (단락): 인접한 두 도체가 의도치 않게 연결된 결함. 에칭 부족이 주원인. 과전류, 발열, 부품 손상 위험.
5. **Spur** (돌기): 도체 패턴에서 불필요하게 돌출된 구리. 에칭 불균일이 원인. 인접 도체와의 단락 위험. spurious_copper와 구별: spur는 기존 패턴에서 돌출.
6. **Spurious Copper** (잔류 구리): 패턴 외 영역에 남은 불필요한 구리 잔여물. 에칭 부족이 주원인. spur와 구별: 독립적으로 존재하는 구리 조각.

## YOLO 탐지 결과

{defects_text}

## 분석 지침

1. **이미지 직접 관찰을 최우선**으로 합니다. YOLO 분류는 참고사항이며, 이미지와 다를 경우 이미지 기반 판단을 우선합니다.
2. YOLO 분류와 실제 이미지가 다른 경우, description에 그 차이와 이유를 명시합니다.
3. **위치 맥락을 분석**합니다 - PCB 가장자리/중심부/패턴 밀집 영역 등 위치가 결함의 의미를 바꿉니다.
4. **복합 결함 상관관계**를 분석합니다 - 여러 결함이 동일 공정 원인(예: 에칭 부족으로 short + spur 동시 발생)을 시사하는지 확인합니다.
5. confidence < 0.7인 결함은 **False Positive 가능성**을 검토합니다. 이미지에서 실제 결함이 보이지 않으면 FP로 판단합니다.
6. 크롭 이미지의 해상도가 낮으면, **전체 이미지(1번째)를 참조**하여 맥락을 파악합니다.

## 심각도 판정 기준 (IPC-A-600 Class 2 기준)

- **critical**: 전기적 기능 불량 직결 (open_circuit, short), 안전 위험
- **high**: 기능에 영향 가능 (missing_hole, 패턴 근접 spur)
- **medium**: 잠재적 위험 (mouse_bite, 독립적 spurious_copper)
- **low**: 외관상 문제만 (미세 spur, 패턴과 먼 잔류 구리)

## 출력 형식

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
모든 텍스트는 **한국어**로 작성하세요.

```json
{{
  "summary": "종합 소견 (2~3문장)",
  "overall_severity": "critical|high|medium|low",
  "defect_analyses": [
    {{
      "defect_index": 0,
      "severity": "critical|high|medium|low",
      "description": "이미지에서 관찰된 결함 외관 설명",
      "possible_cause": "추정 원인 (공정 관점)",
      "recommendation": "권장 조치"
    }}
  ],
  "pattern_analysis": "복합 결함 패턴 분석 (단일 결함이면 null)",
  "quality_score": 0,
  "process_recommendation": "공정 개선 권고사항"
}}
```

quality_score: 0(최악)~100(완벽). 결함 심각도, 개수, 위치를 종합 고려.
"""
    return prompt


async def analyze_defects(image_s3_key: str, detections: List[Dict]) -> dict:
    """
    VLM(Gemini)으로 PCB 결함 분석 수행

    Args:
        image_s3_key: S3 이미지 키
        detections: 결함 목록

    Returns:
        VLM 분석 결과 딕셔너리

    Raises:
        ValueError: API 키 미설정
        RuntimeError: Gemini API 호출 실패
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다")

    # 1. S3에서 이미지 다운로드
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=image_s3_key)
        image_bytes = response["Body"].read()
        logger.info(f"[VLM] S3 이미지 다운로드 완료: {image_s3_key}")
    except Exception as e:
        raise RuntimeError(f"S3 이미지 다운로드 실패: {e}")

    # 2. 어노테이션 이미지 생성
    annotated_bytes = _create_annotated_image(image_bytes, detections)

    # 3. 결함별 크롭 이미지 생성
    crop_images = []
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        crop_bytes = _crop_defect_image(image_bytes, bbox)
        crop_images.append(crop_bytes)

    # 4. 프롬프트 구성
    prompt_text = _build_prompt(detections)

    # 5. Gemini API 요청 구성
    parts = [
        {"text": prompt_text},
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(annotated_bytes).decode("utf-8")
            }
        }
    ]

    # 크롭 이미지 추가
    for crop_bytes in crop_images:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(crop_bytes).decode("utf-8")
            }
        })

    request_body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096
        }
    }

    # 6. Gemini API 호출
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, json=request_body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[VLM] Gemini API HTTP 오류: {e.response.status_code} - {e.response.text[:500]}")
            raise RuntimeError(f"Gemini API 호출 실패 (HTTP {e.response.status_code})")
        except httpx.RequestError as e:
            logger.error(f"[VLM] Gemini API 요청 오류: {e}")
            raise RuntimeError(f"Gemini API 연결 실패: {e}")

    # 7. 응답 파싱
    resp_data = resp.json()

    try:
        text_content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error(f"[VLM] Gemini 응답 파싱 실패: {resp_data}")
        raise RuntimeError("Gemini API 응답 형식이 올바르지 않습니다")

    # JSON 블록 추출 (```json ... ``` 처리)
    json_match = re.search(r"```json\s*(.*?)\s*```", text_content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # ```json 없이 직접 JSON인 경우
        json_str = text_content.strip()

    try:
        analysis = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[VLM] JSON 파싱 실패: {e}\n원본: {text_content[:500]}")
        raise RuntimeError("Gemini 응답을 JSON으로 파싱할 수 없습니다")

    # 8. 메타데이터 병합 (YOLO 탐지 정보를 각 분석 결과에 추가)
    for defect_analysis in analysis.get("defect_analyses", []):
        idx = defect_analysis.get("defect_index", -1)
        if 0 <= idx < len(detections):
            det = detections[idx]
            defect_analysis["defect_type"] = det.get("defect_type")
            defect_analysis["confidence"] = det.get("confidence")
            defect_analysis["bbox"] = det.get("bbox")

    # 9. 분석 메타데이터 추가
    analysis["model"] = "gemini-2.0-flash"
    analysis["analyzed_at"] = datetime.now(KST).isoformat()

    logger.info(f"[VLM] 분석 완료: quality_score={analysis.get('quality_score')}, severity={analysis.get('overall_severity')}")

    return analysis
