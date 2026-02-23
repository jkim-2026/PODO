import os
import wandb
import time
import config

class ModelManager:
    """
    WandB Artifact를 감시하고 최신 모델을 다운로드하는 매니저
    """
    def __init__(self):
        self.api = wandb.Api()
        self.project_path = config.WANDB_PROJECT
        self.artifact_name = config.WANDB_ARTIFACT_NAME
        self.local_model_path = config.MODEL_PATH
        
        # 현재 로컬 모델의 버전(Artifact ID)을 추적하기 위해 별도의 파일 사용
        self.version_file = os.path.join(config.BASE_DIR, ".model_version")
        self.current_version = self._load_current_version()

    def _load_current_version(self):
        if os.path.exists(self.version_file):
            with open(self.version_file, "r") as f:
                return f.read().strip()
        return None

    def _save_current_version(self, version):
        with open(self.version_file, "w") as f:
            f.write(version)
        self.current_version = version

    def _find_pt_in_dir(self, directory):
        """디렉토리에서 첫 번째 .pt 파일 경로를 반환, 없으면 None"""
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".pt"):
                    return os.path.join(root, f)
        return None

    def _local_files_exist(self):
        """실제 .pt 및 .engine 파일이 로컬에 존재하는지 확인"""
        pt_exists     = os.path.exists(self.local_model_path)
        engine_path   = self.local_model_path.replace('.pt', '.engine')
        engine_exists = os.path.exists(engine_path)
        return pt_exists and engine_exists

    def check_for_updates(self):
        """
        WandB에서 업데이트를 확인하고, 새 모델이 있거나 로컬 파일이
        없는 경우 다운로드 → TensorRT 변환 → 배포 준비를 진행합니다.
        """
        try:
            artifact = self.api.artifact(f"{self.project_path}/{self.artifact_name}")
            latest_version = artifact.id

            # 버전이 같고 실제 파일도 존재하면 스킵
            if latest_version == self.current_version and self._local_files_exist():
                return None  # 업데이트 없음

            if latest_version != self.current_version:
                print(f"[ModelManager] 새 모델 버전 발견: {latest_version}. 업데이트 시작...")
            else:
                print(f"[ModelManager] 버전은 같지만 로컬 파일이 없습니다. 재다운로드 시작...")

            # 1. 임시 폴더에 다운로드
            import shutil
            tmp_dir = os.path.join(config.BASE_DIR, "tmp_model_update")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            download_path = artifact.download(root=tmp_dir)

            # .pt 파일 찾기
            pt_file = self._find_pt_in_dir(download_path)

            if not pt_file:
                print("[ModelManager] 에러: .pt 파일을 찾을 수 없습니다.")
                return None

            # 2. 임시 폴더에서 TensorRT 변환 실행 (가장 오래 걸리는 작업)
            print(f"[ModelManager] 젯슨 최적화 시작 (TensorRT 변환)... 이 작업은 수 분이 소요됩니다.")
            from ultralytics import YOLO
            temp_model = YOLO(pt_file)
            temp_model.export(format='engine', dynamic=True, device=0, half=True)

            temp_engine_path = pt_file.replace('.pt', '.engine')
            if not os.path.exists(temp_engine_path):
                print("[ModelManager] 엔진 변환 실패.")
                return None

            # 3. 배포 준비 완료: 파일을 최종 위치로 이동
            final_engine_path = self.local_model_path.replace('.pt', '.engine')
            shutil.copy2(pt_file, self.local_model_path)
            shutil.copy2(temp_engine_path, final_engine_path)

            self._save_current_version(latest_version)
            print(f"[ModelManager] 새 모델 배포 준비 완료: {latest_version}")

            # 임시 폴더 정리
            shutil.rmtree(tmp_dir)

            return final_engine_path

        except Exception as e:
            print(f"[ModelManager] 업데이트 프로세스 중 오류: {e}")
            return None

if __name__ == "__main__":
    # 단독 테스트용
    manager = ModelManager()
    manager.check_for_updates()
