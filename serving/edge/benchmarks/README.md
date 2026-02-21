# Benchmark Runner (M-2)

This directory contains a scenario-based benchmark runner for edge pipeline comparison.

## Files

- `benchmark_runner.py`: runs scenarios and builds summary artifacts
- `scenarios.sample.json`: sample scenario list (edit before running)
- `results/`: output logs and summaries

## Quick Start (Jetson)

Run from `serving/edge`:

```bash
python3 benchmarks/benchmark_runner.py \
  --runner "python3" \
  --scenario-file benchmarks/scenarios.sample.json \
  --var INPUT_FILE=/home/ubuntu/rtsp/test.mp4 \
  --var API_URL=http://<BACKEND_IP>:8080/detect/ \
  --var SESSION_URL=http://<BACKEND_IP>:8080/sessions/
```

If you use `uv`:

```bash
python3 benchmarks/benchmark_runner.py \
  --runner "uv run python" \
  --scenario-file benchmarks/scenarios.sample.json \
  --var INPUT_FILE=/home/ubuntu/rtsp/test.mp4 \
  --var API_URL=http://<BACKEND_IP>:8080/detect/ \
  --var SESSION_URL=http://<BACKEND_IP>:8080/sessions/
```

## Output

Each run creates a timestamped folder under `benchmarks/results/<timestamp>/`.

- `<scenario>/run.log`: raw runtime log
- `summary.json`: parsed metrics per scenario
- `summary.csv`: tabular summary for before/after comparison

## Recommended M-2 comparison flow

1. Checkout baseline commit and run benchmark.
2. Checkout improved commit and run benchmark with same scenario file.
3. Compare the two `summary.csv` files.
4. Focus on:
   - `aggregate_processed_fps`
   - `per_camera_avg_fps`
   - `inference_p95_ms`
   - `upload_wait_p95_ms`, `upload_p95_ms`
   - `max_camera_drop_rate`
   - `queue_drop_frame`, `queue_drop_upload`
