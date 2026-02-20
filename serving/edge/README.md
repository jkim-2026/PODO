# Edge Module (Jetson)

PCB 결함 탐지 시스템의 Edge 실행 모듈입니다.  
역할은 **RTSP 수신 -> PCB 전처리/크롭 -> 추론 -> 업로드/재전송** 입니다.

## 1. 현재 구조 요약

현재 파이프라인은 아래 구조입니다.

```
RTSPReceiver(cam별 thread) -> frame_queue(cam별 독립)
                           -> main 전처리 루프(단일, 라운드로빈 소비)
                           -> shared crop_queue
                           -> InferenceWorker(단일, 모델 1개)
                           -> shared upload_queue
                           -> UploadWorker(단일)
                           -> 실패 시 storage/failed/*.json
                           -> ScavengerWorker(단일, 백그라운드 재전송)
```

핵심 포인트:
- 수신은 camera별 병렬
- 전처리/추론/업로드는 단일 소비자 구조
- 모델은 1개만 로드해 메모리 사용량을 제어
- `camera_id`를 payload에 포함해 채널 구분

## 2. 디렉터리 구조

```text
serving/edge/
├── main.py                # 파이프라인 엔트리
├── config.py              # 실행/큐/임계값 설정
├── preprocessor.py        # PCB 전처리
├── rtsp_receiver.py       # RTSP 수신 스레드
├── inference_worker.py    # 추론 워커 (YOLO/TensorRT)
├── upload_worker.py       # 실시간 업로드 워커
├── scavenger_worker.py    # 실패 파일 재전송 워커
├── metrics.py             # M-1 성능 지표 수집
├── capture_background.py  # 배경 이미지 생성 도구
├── background.png
└── storage/
    └── failed/
        └── expired/       # TTL/재시도 초과 파일
```

## 3. 실행 방법

### 3.1 환경 준비

```bash
cd serving/edge
uv sync --active
```

### 3.2 기본 실행 (1대)

```bash
uv run python main.py
```

### 3.3 멀티 카메라 실행

```bash
uv run python main.py -n 3
```

동작 규칙:
- `-n > 1`이고 입력이 RTSP URL이면 `--input` 뒤에 `_1, _2, ...` 자동 확장
- 예: `rtsp://IP:8554/pcb_stream` -> `pcb_stream_1`, `pcb_stream_2`, `pcb_stream_3`

### 3.4 파일 기반 부하 테스트

```bash
uv run python main.py -i "/path/test.mp4" -n 5
```

동작 규칙:
- 입력이 파일(`.mp4/.avi/.mkv`)이면 같은 파일을 n개 카메라처럼 반복 수신

### 3.5 서버 다운/오프라인 테스트

```bash
uv run python main.py -a "http://127.0.0.1:9999/detect/" --no-session -n 3
```

## 4. CLI 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--input`, `-i` | RTSP URL 또는 비디오 파일 경로 | `config.RTSP_URL` |
| `--api-url`, `-a` | 업로드 API 주소 | `config.API_URL` |
| `--model`, `-m` | YOLO 모델 경로 (`.pt`/`.engine`) | `config.MODEL_PATH` |
| `--loop`, `-l` | 파일 입력 반복 재생 | `False` |
| `--debug`, `-d` | 크롭 이미지 저장 | `False` |
| `--session-url` | 세션 API 주소 | 하드코딩 기본값 |
| `--no-session` | 세션 API 비활성화 | `False` |
| `--no-scavenger` | 재전송 워커 비활성화 | `False` |
| `--max-crops` | 최대 크롭 개수 (0=무제한) | `0` |
| `--num-cameras`, `-n` | 카메라 대수 | `1` |

## 5. 핵심 동작

### 5.1 전처리

- `PCBPreprocessor`는 상태 머신(`background`/`pcb`) 기반
- ROI 표준편차와 배경 차분으로 PCB 영역 추출
- camera별 전처리 인스턴스를 분리해서 상태 오염 방지

### 5.2 추론

- `InferenceWorker`는 모델 1개 로드
- `.pt` 입력 시 `.engine` 존재하면 우선 사용
- 결과 payload에 `camera_id`, `session_id` 포함
- `image_id`는 **마이크로초+UUID**로 생성해 충돌 방지

### 5.3 업로드/재전송

- `UploadWorker`: 실시간 업로드 시도
- 실패 시 `storage/failed/*.json` 저장
- `ScavengerWorker`: 실패 파일을 백그라운드 재전송
  - exponential backoff + jitter
  - max retry / TTL
  - 초과 시 `storage/failed/expired`로 이동

## 6. M-1 성능 지표 (5초 주기 + 종료 요약)

### 6.1 출력되는 지표

1. 처리량
- `throughput(total): input/processed/inference`
- `총 처리량(Aggregate Processed FPS)` (전체 합산)
- `카메라당 평균 처리 FPS(참고)`

2. 카메라별 상태
- `input fps`
- `processed fps`
- `drop rate`

3. 업로드 전달률
- `upload(live)` 실시간 업로드 성공/실패
- `delivery(effective)` = 실시간 + 재전송 성공률
- `backlog` 실패 파일 잔량

4. 큐 지표
- `queue_hwm` (frame/crop/upload 최고 수위)
- `queue_drop(frame/crop/upload)` 누적 드롭 수

5. 지연 지표 (`p50/p95`, ms)
- `frame_wait`, `preprocess`, `crop_wait`, `inference`
- `upload_wait`, `upload`, `e2e`

6. 자동 진단
- `diagnosis`: 병목 의심 구간 요약

### 6.2 해석 가이드

- `Aggregate Processed FPS`는 카메라 합산 처리량입니다.
- 카메라별 실시간성 판단은 `processed fps`와 `drop rate`로 확인합니다.
- 업로드 병목은 보통 아래 신호로 드러납니다.
  - `upload queue_hwm`이 최대치 근접
  - `upload_wait p95` 급증
  - `upload p95` 급증
  - `delivery(effective)` 저하 또는 backlog 증가

## 7. 주요 설정 (`config.py`)

큐/계측/재전송 관련 주요 파라미터:

- `FRAME_QUEUE_SIZE`, `CROP_QUEUE_SIZE`, `UPLOAD_QUEUE_SIZE`
- `METRICS_LOG_INTERVAL_SEC`, `METRICS_LATENCY_BUFFER_SIZE`
- `SCAVENGER_*` (`POLL_INTERVAL`, `BASE_BACKOFF`, `MAX_BACKOFF`, `JITTER`, `MAX_RETRIES`, `TTL`)

## 8. 현재 한계 (정직한 상태 공유)

- 전처리/추론/업로드는 여전히 단일 소비자 구조
- RTSP는 OpenCV FFMPEG 경로 기준 (HW decode 전환 전)

즉 현재는 카메라별 frame queue 분리 + 공정 소비는 적용됐지만, 업로드 구간 병목 영향은 여전히 큽니다.

## 9. 다음 개선 우선순위

1. P1-E1 GStreamer + `nvv4l2decoder` HW decode
2. P1-2 큐 스케일링/경보 고도화
3. P1-4 RTSP 재연결 강화
4. P1-E2 GPU 전처리 POC
5. M-2 Before/After 벤치마크 리포트
