import os

# RTSP 및 백엔드 서버 설정
RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"
API_URL = "http://3.35.182.98:8080/detect/"

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
BACKGROUND_PATH = os.path.join(BASE_DIR, "background.png")

# 전처리 파라미터 (ROI 및 임계값)
# 해상도 1920x1080 기준
ROI_X1, ROI_X2 = 950, 970
ROI_Y1, ROI_Y2 = 158, 922

THRESH_LOW = 15
THRESH_HIGH = 25
BG_DIFF_THRESH = 25

# PCB 크기 검증 파라미터
MIN_HEIGHT, MAX_HEIGHT = 750, 780
MIN_WIDTH, MAX_WIDTH = 700, 1600
MIN_RATIO, MAX_RATIO = 0.8, 2.0

# 시스템 설정
FRAME_QUEUE_SIZE = 2
CROP_QUEUE_SIZE = 10
UPLOAD_QUEUE_SIZE = 20

# 성능 계측 설정
METRICS_LOG_INTERVAL_SEC = 5
METRICS_LATENCY_BUFFER_SIZE = 20000

# Scavenger 재전송 설정
SCAVENGER_ENABLED = True
SCAVENGER_POLL_INTERVAL_SEC = 2
SCAVENGER_BASE_BACKOFF_SEC = 1
SCAVENGER_MAX_BACKOFF_SEC = 60
SCAVENGER_JITTER_RATIO = 0.2
SCAVENGER_MAX_RETRIES = 20
SCAVENGER_TTL_SEC = 24 * 60 * 60

# 저장 경로
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
FAILED_DIR = os.path.join(STORAGE_DIR, "failed")
EXPIRED_DIR = os.path.join(FAILED_DIR, "expired")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_crops")
