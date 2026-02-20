"""
Scavenger worker.

Retries failed upload JSON files in the background with exponential backoff.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import datetime
from typing import Dict

import requests

import config


class ScavengerWorker(threading.Thread):
    """Background retry worker for files in storage/failed."""

    def __init__(
        self,
        failed_dir: str,
        api_url: str,
        metrics=None,
        poll_interval_sec: float = 2.0,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 60.0,
        jitter_ratio: float = 0.2,
        max_retries: int = 20,
        ttl_sec: float = 24 * 60 * 60,
    ):
        super().__init__(daemon=True)
        self.failed_dir = failed_dir
        self.api_url = api_url
        self.metrics = metrics
        self.poll_interval_sec = poll_interval_sec
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.jitter_ratio = jitter_ratio
        self.max_retries = max_retries
        self.ttl_sec = ttl_sec
        self.running = False

        self._retry_state: Dict[str, Dict] = {}
        self._stop_event = threading.Event()

        self.expired_dir = os.path.join(self.failed_dir, "expired")
        os.makedirs(self.failed_dir, exist_ok=True)
        os.makedirs(self.expired_dir, exist_ok=True)

    def run(self):
        self.running = True
        print("[ScavengerWorker] 재전송 모니터링 시작...")

        while not self._stop_event.is_set():
            self._scan_and_retry()
            self._stop_event.wait(self.poll_interval_sec)

        self.running = False
        print("[ScavengerWorker] 종료")

    def _scan_and_retry(self):
        try:
            candidates = [
                os.path.join(self.failed_dir, name)
                for name in os.listdir(self.failed_dir)
                if name.endswith(".json")
            ]
        except FileNotFoundError:
            return

        candidates.sort(key=lambda path: os.path.getmtime(path))
        now = time.time()

        for path in candidates:
            if not os.path.exists(path):
                continue

            state = self._retry_state.get(path)
            if state is None:
                state = {
                    "retry_count": 0,
                    "first_seen": now,
                    "next_retry": now,
                }
                self._retry_state[path] = state

            if now < state["next_retry"]:
                continue

            if (now - state["first_seen"]) > self.ttl_sec or state["retry_count"] >= self.max_retries:
                self._expire_file(path, state)
                continue

            payload = self._read_payload(path)
            if payload is None:
                self._expire_file(path, state)
                continue

            start = time.time()
            ok = self._post_payload(payload)
            upload_ms = (time.time() - start) * 1000.0

            if self.metrics:
                self.metrics.record_latency("upload_ms", upload_ms)
                self.metrics.record_upload_result(ok, source="scavenger")

            if ok:
                camera_id = payload.get("camera_id", "unknown")
                print(f"[ScavengerWorker][{camera_id}] 재전송 성공: {os.path.basename(path)}")
                try:
                    os.remove(path)
                except OSError:
                    pass
                self._retry_state.pop(path, None)
                continue

            state["retry_count"] += 1
            backoff = min(self.base_backoff_sec * (2 ** (state["retry_count"] - 1)), self.max_backoff_sec)
            jitter = backoff * self.jitter_ratio * random.random()
            state["next_retry"] = now + backoff + jitter

    def _read_payload(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ScavengerWorker] 파일 읽기 실패: {path} ({e})")
            return None

    def _post_payload(self, payload: dict) -> bool:
        api_key = os.getenv("EDGE_API_KEY")
        headers = {"X-API-KEY": api_key} if api_key else {}
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
            return response.status_code in (200, 201)
        except requests.exceptions.RequestException:
            return False

    def _expire_file(self, path: str, state: Dict):
        base = os.path.basename(path)
        expired_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base}"
        expired_path = os.path.join(self.expired_dir, expired_name)

        try:
            os.replace(path, expired_path)
            print(
                f"[ScavengerWorker] 재전송 만료 이동: {base} "
                f"(retry={state['retry_count']}, age={int(time.time() - state['first_seen'])}s)"
            )
        except Exception as e:
            print(f"[ScavengerWorker] 만료 파일 이동 실패: {base} ({e})")
        finally:
            self._retry_state.pop(path, None)
            if self.metrics:
                self.metrics.record_scavenger_expired()

    def stop(self):
        self._stop_event.set()
