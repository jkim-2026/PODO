import threading
import queue
import requests
import json
import os
import time
from datetime import datetime
from uuid import uuid4
import config

class UploadWorker(threading.Thread):
    """
    백엔드 서버로 추론 결과를 비동기 업로드하는 워커
    전송 실패 시 로컬 저장을 통해 데이터 손실 방지 (Failover)
    """

    def __init__(self, upload_queue: queue.Queue, metrics=None, api_url: str = None):
        super().__init__(daemon=True)
        self.upload_queue = upload_queue
        self.metrics = metrics
        self.api_url = api_url or config.API_URL
        self.running = False
        
        # 저장 경로 생성
        os.makedirs(config.FAILED_DIR, exist_ok=True)

    def run(self):
        self.running = True
        print(f"[UploadWorker] 업로드 큐 모니터링 시작...")
        
        while self.running:
            try:
                # 큐에서 페이로드/메타 가져오기
                item = self.upload_queue.get(timeout=1.0)
                payload = item
                meta = {}
                if isinstance(item, dict) and "payload" in item:
                    payload = item.get("payload", {})
                    meta = item.get("meta", {})

                if not isinstance(payload, dict):
                    self.upload_queue.task_done()
                    continue

                image_id = payload.get("image_id", "unknown")
                camera_id = payload.get("camera_id", meta.get("camera_id", "unknown"))
                
                # 백엔드 전송 시도
                if self.metrics and meta.get("inference_done_ts"):
                    self.metrics.record_latency(
                        "upload_queue_wait_ms",
                        (time.time() - meta["inference_done_ts"]) * 1000.0,
                    )

                start = time.time()
                ok, status_code, error_text = self._post_payload(payload)
                upload_ms = (time.time() - start) * 1000.0
                if self.metrics:
                    self.metrics.record_latency("upload_ms", upload_ms)
                    self.metrics.record_upload_result(ok, source="live")

                if ok:
                    print(f"[UploadWorker][{camera_id}] {image_id} 전송 성공!")
                    frame_ts = meta.get("frame_ts")
                    if frame_ts and self.metrics:
                        self.metrics.record_latency("e2e_ms", (time.time() - frame_ts) * 1000.0)
                else:
                    if status_code is not None:
                        print(f"[UploadWorker][{camera_id}] {image_id} 전송 실패 (HTTP {status_code}): {error_text}")
                    else:
                        print(f"[UploadWorker][{camera_id}] {image_id} 서버 연결 실패: {error_text}")
                    self._save_locally(payload, image_id)
                
                self.upload_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[UploadWorker] 루프 오류: {e}")

    def _post_payload(self, payload):
        api_key = os.getenv("EDGE_API_KEY")
        headers = {}
        if api_key:
            headers["X-API-KEY"] = api_key

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
            if response.status_code in [200, 201]:
                return True, response.status_code, ""
            return False, response.status_code, response.text
        except requests.exceptions.RequestException as e:
            return False, None, str(e)

    def _save_locally(self, payload, image_id):
        """전송 실패한 데이터를 로컬에 JSON으로 저장"""
        # 파일명 충돌 방지를 위해 마이크로초 + UUID를 추가
        safe_image_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_id)
        filename = (
            f"{safe_image_id}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid4().hex[:8]}.json"
        )
        filepath = os.path.join(config.FAILED_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[UploadWorker] 데이터 로컬 저장 완료: {filepath}")
        except Exception as e:
            print(f"[UploadWorker] 로컬 저장 실패: {e}")

    def stop(self):
        self.running = False
        print("[UploadWorker] 중지 중...")
