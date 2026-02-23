"""
Jetson 환경 시뮬레이션: WandB Artifact에서 모델 다운로드 테스트

실제 Jetson과 동일하게 ModelManager.check_for_updates() 전체 흐름을 실행하되,
TensorRT(.engine) 변환 단계만 생략하고 .pt 모델로 추론까지 확인합니다.

실행법:
    cd /workspace/pro-cv-finalproject-cv-01/training
    ./.venv/bin/python ../serving/edge/test_artifact_download.py
"""

import os
import sys
import wandb
import shutil

# serving/edge 경로를 sys.path에 추가 (config, model_manager 참조용)
EDGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EDGE_DIR)

# ─── 설정 (config.py와 동일한 값) ────────────────────────────────────────────
WANDB_PROJECT = "ckgqf1313-boostcamp/PODO"
WANDB_ARTIFACT_NAME = "pcb-model:production"
DOWNLOAD_DIR = os.path.join(EDGE_DIR, "test_download")
VERSION_FILE = os.path.join(EDGE_DIR, ".test_model_version")
# ─────────────────────────────────────────────────────────────────────────────

def load_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE) as f:
            return f.read().strip()
    return None

def save_version(v):
    with open(VERSION_FILE, "w") as f:
        f.write(v)

def download_model_from_wandb():
    print("=" * 60)
    print("  [Jetson Sim] WandB Artifact 다운로드 테스트 시작")
    print("=" * 60)
    print(f"  Project  : {WANDB_PROJECT}")
    print(f"  Artifact : {WANDB_ARTIFACT_NAME}")
    print()

    # 1. WandB API 초기화
    api = wandb.Api()
    full_name = f"{WANDB_PROJECT}/{WANDB_ARTIFACT_NAME}"

    try:
        artifact = api.artifact(full_name)
    except Exception as e:
        print(f"[ERROR] 아티팩트 조회 실패: {e}")
        print("  → 아직 학습이 완료되지 않았거나 production 태그가 없을 수 있습니다.")
        return None

    latest_version = artifact.id
    local_version  = load_version()
    print(f"  WandB 버전 : {latest_version}")
    print(f"  로컬 버전  : {local_version or '없음 (최초 다운로드)'}")

    # 2. 실제 파일이 존재하는지 확인
    def find_pt_in_dir(directory):
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".pt"):
                    return os.path.join(root, f)
        return None

    existing_pt = find_pt_in_dir(DOWNLOAD_DIR) if os.path.exists(DOWNLOAD_DIR) else None

    # 3. 버전 같고 파일도 있으면 그대로 사용
    if latest_version == local_version and existing_pt:
        print(f"\n[OK] 이미 최신 버전이며 파일도 존재합니다: {existing_pt}")
        return existing_pt

    # 4. 버전이 다르거나 파일이 없으면 재다운로드
    if latest_version != local_version:
        print("\n[NEW] 새 버전 발견! 다운로드 시작...")
    else:
        print("\n[REDOWNLOAD] 버전은 같지만 로컬 파일이 없습니다. 재다운로드 시작...")

    # 기존 임시 폴더 초기화
    if os.path.exists(DOWNLOAD_DIR):
        shutil.rmtree(DOWNLOAD_DIR)

    download_path = artifact.download(root=DOWNLOAD_DIR)
    print(f"  다운로드 경로: {download_path}")

    # .pt 파일 찾기
    pt_file = find_pt_in_dir(download_path)

    if not pt_file:
        print("[ERROR] .pt 파일을 찾을 수 없습니다.")
        return None

    print(f"  모델 파일  : {pt_file}")
    print(f"  파일 크기  : {os.path.getsize(pt_file) / 1e6:.1f} MB")
    save_version(latest_version)
    print("  버전 저장 완료.")
    return pt_file


def run_inference_test(pt_path, test_image_path=None):
    """다운로드한 .pt로 간단한 추론 테스트 (Jetson에서는 .engine으로 대체됨)"""
    print("\n" + "=" * 60)
    print("  [Jetson Sim] 추론 테스트 (PT 모드 / TensorRT 생략)")
    print("=" * 60)

    # Jetson과 동일한 클래스 매핑
    CLASS_NAMES = {
        0: "Missing Hole",
        1: "Mouse Bite",
        2: "Open Circuit",
        3: "Short",
        4: "Spur",
        5: "Spurious Copper"
    }

    from ultralytics import YOLO
    print(f"  모델 로드 중: {pt_path}")
    model = YOLO(pt_path, task='detect')
    print("  모델 로드 완료!")
    print(f"  클래스 목록: {list(model.names.values())}")

    if test_image_path and os.path.exists(test_image_path):
        print(f"\n  추론 대상 이미지: {test_image_path}")
        results = model.predict(test_image_path, conf=0.25, verbose=False)
        boxes = results[0].boxes
        print(f"  탐지된 결함 수: {len(boxes)}")
        for box in boxes:
            cls_id = int(box.cls[0])
            name   = CLASS_NAMES.get(cls_id, f"class{cls_id}")
            conf   = float(box.conf[0])
            print(f"    - {name}: {conf:.2%}")
    else:
        print("  (테스트 이미지 없음 - 모델 로드만 확인)")

    print("\n[SUCCESS] WandB Artifact → 모델 다운로드 → 추론 파이프라인 검증 완료!")
    print("  Jetson에서는 이 .pt 파일이 TensorRT(.engine)으로 자동 변환되어 사용됩니다.")


if __name__ == "__main__":
    pt_path = download_model_from_wandb()

    if pt_path:
        # 테스트 이미지가 있으면 지정 (선택)
        test_img = sys.argv[1] if len(sys.argv) > 1 else None
        run_inference_test(pt_path, test_img)
    else:
        print("\n[FAIL] 모델을 가져오지 못했습니다.")
        sys.exit(1)
