from ultralytics import YOLO
import os

def get_model(config):
    """
    YOLO 모델 인스턴스를 반환합니다.
    구체적인 가중치 파일은 내부 로직 또는 설정(config)에 의해 결정됩니다.
    기본값: yolov8m.pt
    """
    # 이 모듈의 기본 모델 정의
    model_name = "yolov8m.pt"
    
    # 1. 파일이 현재 경로에 직접 존재하는지 확인
    if os.path.exists(model_name):
        pass # 그대로 사용
    # 2. pretrained_weights 폴더 내부에 존재하는지 확인
    elif os.path.exists(os.path.join("pretrained_weights", model_name)):
        model_name = os.path.join("pretrained_weights", model_name)
    
    # 3. 모델 초기화
    model = YOLO(model_name)
    return model
