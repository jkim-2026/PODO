# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCB 결함 탐지 Edge AI 시스템의 **백엔드 서버**.

- **역할**: 엣지(Jetson)에서 추론 결과를 받아 저장하고, 프론트(Streamlit)에 통계/이력 제공
- **기간**: 1개월 (마감 2월 11일)
- **팀**: 5명 중 백엔드 담당

```
[Jetson/엣지] ──POST /detect──→ [FastAPI/백엔드] ←──GET /stats── [Streamlit/프론트]
```

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLite (개발 초기엔 메모리/JSON도 가능)
- **Image Storage**: 파일 시스템 (/images/defects/)
- **Python**: >=3.11

## API Endpoints

### POST /detect (엣지 → 백엔드)

엣지에서 추론 결과 수신

**Request Body:**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| timestamp | string | O | ISO 8601 형식 ("2026-01-14T15:30:45") |
| image_id | string | O | 이미지 식별자 ("PCB_001234") |
| result | string | O | "normal" 또는 "defect" |
| confidence | float | O | 신뢰도 (0.0~1.0) |
| defect_type | string | X | 결함 종류 (불량일 때만) |
| bbox | list[int] | X | [x1, y1, x2, y2] (불량일 때만) |
| image | string | X | Base64 인코딩 이미지 (불량일 때만) |

**Response:**

```json
{"status": "ok", "id": 12345}
```

### GET /stats (프론트 → 백엔드)

통계 정보 반환

**Response:**

| 필드 | 타입 | 설명 |
|------|------|------|
| total_count | int | 총 검사 수 |
| normal_count | int | 정상 개수 |
| defect_count | int | 불량 개수 |
| defect_rate | float | 불량률 (%) |
| avg_fps | float | 평균 FPS |
| last_defect | object | 가장 최근 불량 정보 |

**last_defect 객체:**

| 필드 | 타입 | 설명 |
|------|------|------|
| timestamp | string | 검사 시각 |
| image_id | string | 이미지 ID |
| result | string | "defect" |
| confidence | float | 신뢰도 |
| defect_type | string | 결함 종류 |
| bbox | list[int] | 결함 위치 |
| image_path | string | 이미지 경로 |

### GET /latest

최근 검사 이력 10개 반환

### GET /defects

결함 타입별 집계 반환

## Data Schema

**검사 로그 테이블 (inspection_logs):**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PRIMARY KEY |
| timestamp | TEXT | 검사 시각 |
| image_id | TEXT | 이미지 식별자 |
| result | TEXT | normal/defect |
| confidence | REAL | 신뢰도 |
| defect_type | TEXT | 결함 종류 (nullable) |
| bbox | TEXT | JSON 문자열 (nullable) |
| image_path | TEXT | 이미지 저장 경로 (nullable) |

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install fastapi uvicorn python-multipart aiosqlite
```

## Running the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 문서 확인: http://localhost:8000/docs

## Project Structure

```
├── main.py              # FastAPI 앱 진입점, 라우터 등록
├── routers/
│   ├── detect.py        # POST /detect 엔드포인트
│   └── stats.py         # GET /stats, /latest, /defects
├── models/
│   └── schemas.py       # Pydantic 요청/응답 스키마
├── database/
│   └── db.py            # SQLite 연결 및 CRUD 함수
├── images/
│   └── defects/         # 불량 이미지 저장 폴더
├── pyproject.toml
└── CLAUDE.md
```

## Implementation Priority

1. **1순위**: POST /detect + 메모리 저장 (엣지 연동 테스트용)
2. **2순위**: GET /stats 기본 구현 (프론트 연동용)
3. **3순위**: SQLite 전환 + 이미지 파일 저장
4. **4순위**: GET /latest, GET /defects 추가
5. **5순위**: 에러 핸들링, 로깅

## Important Notes

- **CORS 설정 필수**: 프론트엔드 도메인에서 API 호출 허용
- **이미지 저장**: DB에 직접 저장 X → 파일 시스템에 저장, DB엔 경로만
- **Base64 디코딩**: 불량 이미지 수신 시 디코딩해서 파일로 저장
- **Static Files**: 저장된 이미지를 프론트에서 불러올 수 있게 서빙

## Collaboration

**엣지 담당과 협의:**
- 전송 주기 : 매 프레임
- 비동기 전송
- 전송 실패 시 처리 방식 : 데이터 유실

**프론트 담당과 협의:**
- 폴링 주기 : 현재 1초
- 추가로 필요한 통계 데이터 : 향후 정함
- 시간별/일별 그래프용 데이터 필요 여부 : 향후 정함

**코드 주석은 한국어로 작성해줘**

## Commit Convention

커밋 메시지 작성 시 `.gitmessage` 템플릿을 따를 것.

**형식:**
```
[파트] 타입: 제목 (50자 이내)

왜 변경했나요? (선택사항)

특이사항/영향받는 부분 (선택사항)
```

**파트:**
- `BE` : 백엔드 (FastAPI)
- `Edge` : 엣지/추론 (Jetson)
- `FE` : 프론트엔드 (Streamlit)
- `Train` : 모델 학습
- `Docs` : 문서
- `Config` : 설정/환경

**타입:**
- `feat` : 새 기능 추가
- `fix` : 버그 수정
- `refactor` : 코드 리팩토링
- `test` : 테스트 추가/수정
- `docs` : 문서 수정
- `style` : 코드 포맷팅
- `chore` : 빌드, 설정 파일 수정

**예시:**
```
[BE] feat: POST /detect 엔드포인트 구현
[Edge] fix: Base64 인코딩 오류 수정
[FE] feat: 실시간 그래프 추가
```