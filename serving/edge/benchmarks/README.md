# Benchmark Runner (M-2)

This directory contains a scenario-based benchmark runner for edge pipeline comparison.

이 디렉토리는 엣지 파이프라인의 Before/After 비교를 위한 시나리오 기반 벤치마크 도구입니다.

## Files / 파일 구성

- `benchmark_runner.py`: runs scenarios and builds summary artifacts  
  시나리오를 순차 실행하고 결과 요약 파일(`summary.json`, `summary.csv`)을 생성합니다.
- `scenarios.sample.json`: sample scenario list (edit before running)  
  실행할 시나리오 샘플 목록입니다. 환경에 맞게 수정해서 사용합니다.
- `results/`: output logs and summaries  
  실행 로그와 요약 결과가 저장되는 디렉토리입니다.

## Quick Start (Jetson) / 빠른 실행

Run from `serving/edge`:

`serving/edge` 경로에서 실행합니다.

```bash
python3 benchmarks/benchmark_runner.py \
  --runner "python3" \
  --scenario-file benchmarks/scenarios.sample.json
```

If you use `uv`:

`uv`를 사용하는 경우:

```bash
python3 benchmarks/benchmark_runner.py \
  --runner "uv run python" \
  --scenario-file benchmarks/scenarios.sample.json
```

기본 시나리오는 `main.py`의 기본값을 사용합니다.
- RTSP 입력: `serving/edge/config.py`의 `RTSP_URL`
- 백엔드 API: `serving/edge/config.py`의 `API_URL`

## Scenario Format / 시나리오 형식

`scenarios.sample.json`의 각 항목은 아래 필드를 가집니다.

- `name`: 시나리오 이름 (폴더명/요약 키로 사용)
- `description`: 설명
- `duration_sec`: 실행 시간(초)
- `args`: `main.py` 실행 인자 배열
- `enabled` (optional): `false`면 스킵

필요하면 `--var KEY=VALUE`로 `${KEY}` 템플릿 값을 치환할 수 있습니다.

## Useful Options / 자주 쓰는 옵션

- `--only <name>`: 특정 시나리오만 실행 (여러 번 지정 가능)
- `--dry-run`: 실제 실행 없이 명령만 출력
- `--stop-grace-sec <sec>`: 종료 시 SIGINT 이후 대기 시간
- `--workdir <path>`: 실행 작업 디렉토리

## Output / 결과물

Each run creates a timestamped folder under `benchmarks/results/<timestamp>/`.

매 실행마다 `benchmarks/results/<timestamp>/` 폴더가 생성됩니다.

- `<scenario>/run.log`: raw runtime log  
  각 시나리오의 원본 실행 로그
- `summary.json`: parsed metrics per scenario  
  시나리오별 파싱 결과(JSON)
- `summary.csv`: tabular summary for before/after comparison  
  비교용 표 형태 요약(CSV)

## Recommended M-2 comparison flow / 권장 비교 절차

1. Checkout baseline commit and run benchmark.  
   기준(Before) 커밋에서 벤치마크 실행
2. Checkout improved commit and run benchmark with same scenario file.  
   개선(After) 커밋에서 동일 시나리오로 재실행
3. Compare the two `summary.csv` files.  
   두 실행의 `summary.csv` 비교
4. Focus on / 중점 지표:
   - `aggregate_processed_fps`
   - `per_camera_avg_fps`
   - `inference_p95_ms`
   - `upload_wait_p95_ms`, `upload_p95_ms`
   - `max_camera_drop_rate`
   - `queue_drop_frame`, `queue_drop_upload`
