"""
PCB 전처리 파이프라인 메인 모듈

RTSP 영상을 수신하고, PCB를 감지/크롭하여 추론 및 업로드 수행
"""

import argparse
import os
import queue
import signal
import sys
import time
from datetime import datetime
from typing import Dict

import cv2
import requests

import config
from metrics import EdgeMetrics, format_ms
from preprocessor import PCBPreprocessor
from rtsp_receiver import RTSPReceiver
from scavenger_worker import ScavengerWorker
from inference_worker import InferenceWorker
from upload_worker import UploadWorker


def _dict_delta(current: Dict[str, int], previous: Dict[str, int]) -> Dict[str, int]:
    keys = set(current.keys()) | set(previous.keys())
    return {key: current.get(key, 0) - previous.get(key, 0) for key in keys}


def _print_periodic_metrics(metrics: EdgeMetrics, previous_snapshot: dict) -> dict:
    snapshot = metrics.snapshot(config.FAILED_DIR)
    interval = max(snapshot["ts"] - previous_snapshot["ts"], 1e-6)

    delta_input = _dict_delta(snapshot["input_frames"], previous_snapshot["input_frames"])
    delta_processed = _dict_delta(snapshot["processed_frames"], previous_snapshot["processed_frames"])
    delta_inference = _dict_delta(snapshot["inference_count"], previous_snapshot["inference_count"])

    print("\n[Metrics] === 5s Summary ===")
    for camera_id in sorted(set(delta_input.keys()) | set(delta_processed.keys()) | set(delta_inference.keys())):
        input_fps = delta_input.get(camera_id, 0) / interval
        proc_fps = delta_processed.get(camera_id, 0) / interval
        inf_fps = delta_inference.get(camera_id, 0) / interval
        total_input = snapshot["input_frames"].get(camera_id, 0)
        total_drop = snapshot["input_drops"].get(camera_id, 0)
        drop_rate = (total_drop / total_input * 100.0) if total_input > 0 else 0.0
        print(
            f"  [{camera_id}] input={input_fps:.2f}fps, processed={proc_fps:.2f}fps, "
            f"infer={inf_fps:.2f}fps, drop={drop_rate:.2f}% ({total_drop}/{total_input})"
        )

    total_infer_fps = (snapshot["totals"]["inference"] - previous_snapshot["totals"]["inference"]) / interval
    print(
        f"  throughput={total_infer_fps:.2f} crops/sec, "
        f"upload_success={snapshot['upload_success']}, upload_fail={snapshot['upload_fail']}, "
        f"backlog={snapshot['failed_backlog']}"
    )
    print(
        f"  scavenger: success={snapshot['scavenger_success']}, fail={snapshot['scavenger_fail']}, "
        f"expired={snapshot['scavenger_expired']}"
    )
    print(
        "  queue_hwm: "
        f"frame={snapshot['queue_high_watermark'].get('frame_queue', 0)}, "
        f"crop={snapshot['queue_high_watermark'].get('crop_queue', 0)}, "
        f"upload={snapshot['queue_high_watermark'].get('upload_queue', 0)}"
    )
    print(
        "  latency(ms) p50/p95: "
        f"frame_wait={format_ms(snapshot['latency']['frame_queue_wait_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['frame_queue_wait_ms']['p95'])}, "
        f"preprocess={format_ms(snapshot['latency']['preprocess_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['preprocess_ms']['p95'])}, "
        f"crop_wait={format_ms(snapshot['latency']['crop_queue_wait_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['crop_queue_wait_ms']['p95'])}, "
        f"inference={format_ms(snapshot['latency']['inference_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['inference_ms']['p95'])}, "
        f"upload_wait={format_ms(snapshot['latency']['upload_queue_wait_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['upload_queue_wait_ms']['p95'])}, "
        f"upload={format_ms(snapshot['latency']['upload_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['upload_ms']['p95'])}, "
        f"e2e={format_ms(snapshot['latency']['e2e_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['e2e_ms']['p95'])}"
    )
    return snapshot


def start_session(session_url: str) -> int:
    """
    백엔드에 세션 시작을 요청하고 세션 ID를 반환.
    실패 시 None 반환.
    """
    try:
        response = requests.post(session_url, timeout=5.0)
        if response.status_code == 201:
            data = response.json()
            session_id = data.get("id")
            print(f"[Main] 세션 시작: ID={session_id}")
            return session_id
        else:
            print(f"[Main] 세션 시작 실패: HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[Main] 세션 시작 요청 실패: {e}")
        return None


def end_session(session_url: str, session_id: int):
    """
    백엔드에 세션 종료를 요청.
    """
    if session_id is None:
        return

    try:
        response = requests.patch(f"{session_url}{session_id}", timeout=5.0)
        if response.status_code == 200:
            print(f"[Main] 세션 종료: ID={session_id}")
        else:
            print(f"[Main] 세션 종료 실패: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[Main] 세션 종료 요청 실패: {e}")


def save_crop_for_debug(crop, output_dir: str, index: int):
    """디버그용 크롭 이미지 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"crop_{index:04d}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, crop)
    print(f"[Debug] 크롭 저장: {filepath}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="PCB 전처리 및 추론 파이프라인")
    parser.add_argument(
        "--input", "-i",
        default=config.RTSP_URL,
        help=f"RTSP URL 또는 비디오 파일 경로 (기본값: {config.RTSP_URL})"
    )
    parser.add_argument(
        "--api-url", "-a",
        default=config.API_URL,
        help=f"백엔드 API 주소 (기본값: {config.API_URL})"
    )
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL_PATH,
        help=f"YOLO 모델 경로 (기본값: {config.MODEL_PATH})"
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="비디오 파일 반복 재생 (테스트용)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="디버그 모드 (크롭 이미지 저장)"
    )
    parser.add_argument(
        "--session-url",
        default="http://3.35.182.98:8080/sessions/",
        help="세션 API 주소"
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="세션 관리 비활성화"
    )
    parser.add_argument(
        "--no-scavenger",
        action="store_true",
        help="실패 파일 자동 재전송 워커 비활성화"
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="최대 크롭 개수 (0=무제한)"
    )
    parser.add_argument(
        "--num-cameras", "-n",
        type=int,
        default=1,
        help="사용할 카메라 대수 (기본값: 1)"
    )
    
    args = parser.parse_args()

    # 입력 소스 처리
    if args.num_cameras > 1:
        # 단일 주소가 들어오면 자동으로 _1, _2... 를 붙여서 확장
        base_url = args.input
        # 확장자가 있는 파일이 아닌 경우에만 숫자를 붙임
        if not any(base_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv']):
            input_sources = [f"{base_url}_{i+1}" for i in range(args.num_cameras)]
        else:
            # 파일인 경우 동일 파일을 n번 수신 (테스트용)
            input_sources = [base_url] * args.num_cameras
    else:
        # 기존처럼 쉼표로 구분된 입력을 리스트로 변환
        input_sources = [s.strip() for s in args.input.split(',')]

    # 동적 설정 업데이트 (CLI 인자 우선)
    config.API_URL = args.api_url
    config.MODEL_PATH = args.model

    # 배경 이미지 확인
    if not os.path.exists(config.BACKGROUND_PATH):
        print(f"[Main] 에러: 배경 이미지({config.BACKGROUND_PATH})가 없습니다.")
        sys.exit(1)

    # 세션 시작
    session_id = None
    if not args.no_session:
        session_id = start_session(args.session_url)

    # Queue 생성
    frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE * len(input_sources))
    crop_queue = queue.Queue(maxsize=config.CROP_QUEUE_SIZE)
    upload_queue = queue.Queue(maxsize=config.UPLOAD_QUEUE_SIZE)
    metrics = EdgeMetrics(latency_buffer_size=config.METRICS_LATENCY_BUFFER_SIZE)

    # 1. 업로드 워커 시작
    upload_worker = UploadWorker(upload_queue, metrics=metrics, api_url=config.API_URL)
    upload_worker.start()

    # 1-1. 재전송 워커 시작
    scavenger_worker = None
    if config.SCAVENGER_ENABLED and not args.no_scavenger:
        scavenger_worker = ScavengerWorker(
            failed_dir=config.FAILED_DIR,
            api_url=config.API_URL,
            metrics=metrics,
            poll_interval_sec=config.SCAVENGER_POLL_INTERVAL_SEC,
            base_backoff_sec=config.SCAVENGER_BASE_BACKOFF_SEC,
            max_backoff_sec=config.SCAVENGER_MAX_BACKOFF_SEC,
            jitter_ratio=config.SCAVENGER_JITTER_RATIO,
            max_retries=config.SCAVENGER_MAX_RETRIES,
            ttl_sec=config.SCAVENGER_TTL_SEC,
        )
        scavenger_worker.start()

    # 2. 추론 워커 초기화
    print(f"[Main] 추론 워커 초기화 중 (모델: {config.MODEL_PATH})...")
    try:
        inference_worker = InferenceWorker(
            crop_queue,
            upload_queue,
            config.MODEL_PATH,
            session_id=session_id,
            metrics=metrics,
        )
    except Exception as e:
        print(f"[Main] 추론 워커 초기화 실패: {e}")
        upload_worker.stop()
        upload_worker.join(timeout=2.0)
        if scavenger_worker:
            scavenger_worker.stop()
            scavenger_worker.join(timeout=2.0)
        if session_id:
            end_session(args.session_url, session_id)
        sys.exit(1)

    # 3. RTSP 수신 스레드 리스트 시작
    receivers = []
    for i, source in enumerate(input_sources):
        camera_id = f"cam_{i+1}"
        receiver = RTSPReceiver(source, frame_queue, camera_id=camera_id, loop=args.loop, metrics=metrics)
        receiver.start()
        receivers.append(receiver)

    # 연결 대기 (최대 10초)
    print(f"[Main] {len(receivers)}개 RTSP 연결 대기 중...")
    start_wait = time.time()
    while time.time() - start_wait < 10:
        if all(r.is_running() for r in receivers):
            break
        time.sleep(0.1)
    
    # 4. 추론 워커 가동
    inference_worker.start()

    # 카메라별 전처리기 초기화 (상태 오염 방지)
    preprocessors = {
        receiver.camera_id: PCBPreprocessor(config.BACKGROUND_PATH)
        for receiver in receivers
    }

    # 종료 시그널 핸들러
    shutdown_flag = False
    def signal_handler(signum, frame):
        nonlocal shutdown_flag
        print("\n[Main] 종료 신호 수신...")
        shutdown_flag = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Main] 파이프라인 가동 시작")
    print(f"  - 입력 소스: {len(input_sources)}개")
    for r in receivers:
        print(f"    * [{r.camera_id}] {r.source}")
    print(f"  - API: {config.API_URL}")
    print(f"  - 모델: {config.MODEL_PATH}")
    print(f"  - 세션 ID: {session_id if session_id else '없음'}")
    print(f"  - Scavenger: {'활성화' if scavenger_worker else '비활성화'}")

    crop_count = 0
    frame_count = 0
    start_time = time.time()
    last_metrics_ts = start_time
    last_metrics_snapshot = metrics.snapshot(config.FAILED_DIR)

    try:
        while not shutdown_flag:
            try:
                # 큐에서 (camera_id, frame, frame_ts) 튜플을 가져옴
                frame_item = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not r.is_running() for r in receivers):
                    break
                continue

            camera_id = "unknown"
            frame = None
            frame_ts = None

            if isinstance(frame_item, tuple):
                if len(frame_item) == 3:
                    camera_id, frame, frame_ts = frame_item
                elif len(frame_item) >= 2:
                    camera_id, frame = frame_item[0], frame_item[1]
            if frame is None:
                continue

            frame_count += 1
            metrics.record_processed(camera_id)
            metrics.update_queue_depth("frame_queue", frame_queue.qsize())
            if frame_ts:
                metrics.record_latency("frame_queue_wait_ms", (time.time() - frame_ts) * 1000.0)

            preprocessor = preprocessors.get(camera_id)
            if preprocessor is None:
                # 예상치 못한 camera_id가 들어와도 상태를 분리해 처리
                preprocessor = PCBPreprocessor(config.BACKGROUND_PATH)
                preprocessors[camera_id] = preprocessor
                print(f"[Main] 신규 카메라 전처리기 생성: {camera_id}")

            preprocess_start = time.time()
            cropped = preprocessor.process_frame(frame)
            preprocess_ms = (time.time() - preprocess_start) * 1000.0
            metrics.record_latency("preprocess_ms", preprocess_ms)

            if cropped is not None:
                crop_count += 1
                metrics.record_crop(camera_id)
                print(f"[Main][{camera_id}] [#{crop_count}] PCB 포착! ({cropped.shape[1]}x{cropped.shape[0]})")

                crop_item = {
                    "camera_id": camera_id,
                    "crop": cropped,
                    "frame_ts": frame_ts,
                    "preprocess_done_ts": time.time(),
                }
                try:
                    crop_queue.put_nowait(crop_item)
                    metrics.update_queue_depth("crop_queue", crop_queue.qsize())
                except queue.Full:
                    try:
                        crop_queue.get_nowait()
                        metrics.record_queue_drop("crop_queue")
                        crop_queue.put_nowait(crop_item)
                        metrics.update_queue_depth("crop_queue", crop_queue.qsize())
                    except queue.Empty:
                        pass

                if args.debug:
                    save_crop_for_debug(cropped, config.DEBUG_DIR, crop_count)

                if args.max_crops > 0 and crop_count >= args.max_crops:
                    break

            now = time.time()
            if now - last_metrics_ts >= config.METRICS_LOG_INTERVAL_SEC:
                last_metrics_snapshot = _print_periodic_metrics(metrics, last_metrics_snapshot)
                last_metrics_ts = now

    except Exception as e:
        print(f"[Main] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Main] 리소스 정리 중...")
        for r in receivers:
            r.stop()
        inference_worker.stop()
        upload_worker.stop()
        if scavenger_worker:
            scavenger_worker.stop()

        for r in receivers:
            r.join(timeout=2.0)
        inference_worker.join(timeout=2.0)
        upload_worker.join(timeout=2.0)
        if scavenger_worker:
            scavenger_worker.join(timeout=2.0)

        if session_id:
            end_session(args.session_url, session_id)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n[Main] === 최종 통계 ===")
        print(f"  처리 프레임: {frame_count}")
        print(f"  포착된 PCB: {crop_count}")
        print(f"  실행 시간: {elapsed:.1f}초")
        print(f"  평균 성능: {fps:.1f} FPS")
        for r in receivers:
            print(f"  [{r.camera_id}] 상태: {r.get_stats()}")
        print(f"  세션 ID: {session_id if session_id else '없음'}")

        final_snapshot = metrics.snapshot(config.FAILED_DIR)
        print("\n[Metrics] === Final Summary ===")
        for camera_id in sorted(final_snapshot["input_frames"].keys()):
            total_input = final_snapshot["input_frames"].get(camera_id, 0)
            total_drop = final_snapshot["input_drops"].get(camera_id, 0)
            processed = final_snapshot["processed_frames"].get(camera_id, 0)
            inferred = final_snapshot["inference_count"].get(camera_id, 0)
            drop_rate = (total_drop / total_input * 100.0) if total_input > 0 else 0.0
            print(
                f"  [{camera_id}] input={total_input}, processed={processed}, "
                f"inference={inferred}, drop={total_drop} ({drop_rate:.2f}%)"
            )

        print(
            f"  upload_success={final_snapshot['upload_success']}, "
            f"upload_fail={final_snapshot['upload_fail']}, "
            f"upload_success_rate={final_snapshot['upload_success_rate']:.2f}%, "
            f"backlog={final_snapshot['failed_backlog']}"
        )
        print(
            f"  scavenger_success={final_snapshot['scavenger_success']}, "
            f"scavenger_fail={final_snapshot['scavenger_fail']}, "
            f"scavenger_expired={final_snapshot['scavenger_expired']}"
        )
        print(
            "  queue_hwm: "
            f"frame={final_snapshot['queue_high_watermark'].get('frame_queue', 0)}, "
            f"crop={final_snapshot['queue_high_watermark'].get('crop_queue', 0)}, "
            f"upload={final_snapshot['queue_high_watermark'].get('upload_queue', 0)}"
        )
        print(
            "  latency(ms) p50/p95: "
            f"frame_wait={format_ms(final_snapshot['latency']['frame_queue_wait_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['frame_queue_wait_ms']['p95'])}, "
            f"preprocess={format_ms(final_snapshot['latency']['preprocess_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['preprocess_ms']['p95'])}, "
            f"crop_wait={format_ms(final_snapshot['latency']['crop_queue_wait_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['crop_queue_wait_ms']['p95'])}, "
            f"inference={format_ms(final_snapshot['latency']['inference_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['inference_ms']['p95'])}, "
            f"upload_wait={format_ms(final_snapshot['latency']['upload_queue_wait_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['upload_queue_wait_ms']['p95'])}, "
            f"upload={format_ms(final_snapshot['latency']['upload_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['upload_ms']['p95'])}, "
            f"e2e={format_ms(final_snapshot['latency']['e2e_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['e2e_ms']['p95'])}"
        )


if __name__ == "__main__":
    main()
