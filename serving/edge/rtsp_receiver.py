"""
RTSP 수신 스레드

RTSP 스트림 또는 비디오 파일에서 프레임을 읽어 Queue에 전달
"""

import os
import re
import time
import cv2
import queue
import threading
from typing import Optional, Tuple

# RTSP TCP 모드 설정 (UDP 패킷 손실 방지)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPReceiver(threading.Thread):
    """
    RTSP 수신 스레드

    백그라운드에서 프레임을 읽어 Queue에 넣습니다.
    Queue가 가득 차면 오래된 프레임을 버립니다.
    """

    _gst_supported_cache: Optional[bool] = None

    def __init__(
        self,
        source: str,
        frame_queue: queue.Queue,
        camera_id: str = "default",
        loop: bool = False,
        metrics=None,
        queue_name: str = "frame_queue",
        capture_backend: str = "auto",
        use_hw_decode: bool = True,
        rtsp_latency_ms: int = 120,
        reconnect_enabled: bool = True,
        reconnect_base_delay_sec: float = 1.0,
        reconnect_max_delay_sec: float = 10.0,
        max_failures: int = 30,
    ):
        """
        Args:
            source: RTSP URL 또는 비디오 파일 경로
            frame_queue: 프레임을 넣을 Queue
            camera_id: 카메라 식별자
            loop: 비디오 파일 반복 재생 여부 (테스트용)
        """
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.camera_id = camera_id
        self.loop = loop
        self.metrics = metrics
        self.queue_name = queue_name
        self.capture_backend = capture_backend.lower().strip()
        self.use_hw_decode = use_hw_decode
        self.rtsp_latency_ms = max(0, int(rtsp_latency_ms))
        self.reconnect_enabled = reconnect_enabled
        self.reconnect_base_delay_sec = max(0.1, float(reconnect_base_delay_sec))
        self.reconnect_max_delay_sec = max(
            self.reconnect_base_delay_sec,
            float(reconnect_max_delay_sec),
        )
        self.max_failures = max(1, int(max_failures))
        self.active_backend = "none"
        self.running = False
        self._stop_event = threading.Event()

        # 통계
        self.frame_count = 0
        self.drop_count = 0
        self.reconnect_count = 0

    @staticmethod
    def _is_rtsp_source(source: str) -> bool:
        return source.strip().lower().startswith("rtsp://")

    @classmethod
    def _is_gstreamer_supported(cls) -> bool:
        if cls._gst_supported_cache is not None:
            return cls._gst_supported_cache
        try:
            build_info = cv2.getBuildInformation()
            cls._gst_supported_cache = bool(re.search(r"GStreamer\s*:\s*YES", build_info))
        except Exception:
            cls._gst_supported_cache = False
        return cls._gst_supported_cache

    def _build_rtsp_gstreamer_pipeline(self) -> str:
        if self.use_hw_decode:
            decode_chain = (
                "rtph264depay ! h264parse ! "
                "nvv4l2decoder enable-max-performance=1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR"
            )
        else:
            decode_chain = (
                "rtph264depay ! h264parse ! "
                "avdec_h264 ! videoconvert ! video/x-raw,format=BGR"
            )

        return (
            f"rtspsrc location=\"{self.source}\" protocols=tcp latency={self.rtsp_latency_ms} "
            "drop-on-latency=true ! "
            f"{decode_chain} ! "
            "appsink sync=false max-buffers=1 drop=true"
        )

    def _open_with_gstreamer(self) -> Optional[cv2.VideoCapture]:
        if not self._is_rtsp_source(self.source):
            return None
        if not self._is_gstreamer_supported():
            return None
        pipeline = self._build_rtsp_gstreamer_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _open_with_ffmpeg(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화 (실시간성)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _open_capture(self) -> Tuple[Optional[cv2.VideoCapture], str]:
        is_rtsp = self._is_rtsp_source(self.source)
        backend = self.capture_backend
        if backend not in ("auto", "gstreamer", "ffmpeg"):
            backend = "auto"

        if is_rtsp and backend in ("auto", "gstreamer"):
            cap = self._open_with_gstreamer()
            if cap is not None:
                return cap, "gstreamer"
            if backend == "gstreamer":
                print(
                    f"[RTSPReceiver][{self.camera_id}] GStreamer 연결 실패 -> ffmpeg fallback 시도"
                )

        if backend in ("auto", "ffmpeg", "gstreamer"):
            cap = self._open_with_ffmpeg()
            if cap is not None:
                return cap, "ffmpeg"

        return None, "none"

    def _sleep_or_stop(self, sleep_sec: float):
        self._stop_event.wait(max(0.0, sleep_sec))

    def run(self):
        """스레드 메인 루프"""
        print(f"[RTSPReceiver][{self.camera_id}] 연결 시도: {self.source}")

        cap = None
        reconnect_delay_sec = self.reconnect_base_delay_sec
        consecutive_failures = 0

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap, backend_name = self._open_capture()
                if cap is None:
                    if self._is_rtsp_source(self.source) and self.reconnect_enabled:
                        self.running = False
                        self.reconnect_count += 1
                        print(
                            f"[RTSPReceiver][{self.camera_id}] 연결 실패 - "
                            f"{reconnect_delay_sec:.1f}s 후 재시도 ({self.reconnect_count})"
                        )
                        self._sleep_or_stop(reconnect_delay_sec)
                        reconnect_delay_sec = min(
                            reconnect_delay_sec * 2.0,
                            self.reconnect_max_delay_sec,
                        )
                        continue
                    print(f"[RTSPReceiver][{self.camera_id}] 소스를 열 수 없습니다: {self.source}")
                    break

                self.active_backend = backend_name
                reconnect_delay_sec = self.reconnect_base_delay_sec
                consecutive_failures = 0
                self.running = True

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(
                    f"[RTSPReceiver][{self.camera_id}] 연결 성공({self.active_backend}): "
                    f"{width}x{height} @ {fps:.1f}fps"
                )
                print(f"[RTSPReceiver][{self.camera_id}] 수신 시작...")

            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1

                if self.loop and not self._is_rtsp_source(self.source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                    continue

                if consecutive_failures >= self.max_failures:
                    if self._is_rtsp_source(self.source) and self.reconnect_enabled:
                        self.running = False
                        self.reconnect_count += 1
                        print(
                            f"[RTSPReceiver][{self.camera_id}] 프레임 읽기 실패 {consecutive_failures}회 - "
                            f"재연결 시도 ({self.reconnect_count})"
                        )
                        cap.release()
                        cap = None
                        self._sleep_or_stop(reconnect_delay_sec)
                        reconnect_delay_sec = min(
                            reconnect_delay_sec * 2.0,
                            self.reconnect_max_delay_sec,
                        )
                        continue

                    print(
                        f"[RTSPReceiver][{self.camera_id}] 스트림 종료 "
                        f"(연속 {consecutive_failures}회 실패)"
                    )
                    break

                self._sleep_or_stop(0.033)
                continue

            consecutive_failures = 0
            reconnect_delay_sec = self.reconnect_base_delay_sec

            self.frame_count += 1
            if self.metrics:
                self.metrics.record_input(self.camera_id)

            # Queue가 가득 차면 오래된 프레임 버림 (실시간성 우선)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.drop_count += 1
                    if self.metrics:
                        self.metrics.record_input_drop(self.camera_id)
                        self.metrics.record_queue_drop(self.queue_name)
                        if self.queue_name != "frame_queue":
                            self.metrics.record_queue_drop("frame_queue")
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait((self.camera_id, frame, time.time()))
                if self.metrics:
                    self.metrics.update_queue_depth(self.queue_name, self.frame_queue.qsize())
            except queue.Full:
                self.drop_count += 1
                if self.metrics:
                    self.metrics.record_input_drop(self.camera_id)
                    self.metrics.record_queue_drop(self.queue_name)
                    if self.queue_name != "frame_queue":
                        self.metrics.record_queue_drop("frame_queue")

        if cap is not None:
            cap.release()
        self.running = False
        print(
            f"[RTSPReceiver][{self.camera_id}] 종료 - 총 프레임: {self.frame_count}, "
            f"드롭: {self.drop_count}, 재연결: {self.reconnect_count}"
        )

    def stop(self):
        """스레드 정지 요청"""
        self._stop_event.set()

    def is_running(self) -> bool:
        """실행 중 여부"""
        return self.running

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            "frame_count": self.frame_count,
            "drop_count": self.drop_count,
            "running": self.running,
            "backend": self.active_backend,
            "reconnect_count": self.reconnect_count,
        }
