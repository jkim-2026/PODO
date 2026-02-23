import os
import sys
import yaml
import argparse
import sys
import os

# 프로젝트 루트 디렉토리를 경로에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import socket
# [몽키 패치] Ultralytics 오프라인 실행 시 발생하는 네트워크 대기(hang) 문제를 우회합니다.
_original_getaddrinfo = socket.getaddrinfo
def _dummy_getaddrinfo(*args, **kwargs):
    # 속도를 위해 오프라인 상태라고 가정하고 즉각적으로 오류를 발생시킵니다.
    raise socket.gaierror("속도 향상을 위한 가상 오프라인 상태")
socket.getaddrinfo = _dummy_getaddrinfo

# Ultralytics 모듈 임포트
try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
finally:
    # 몽키 패치 원상 복구
    socket.getaddrinfo = _original_getaddrinfo

# 사용자 정의 QAT 설정 유틸리티 임포트
from src.qat import utils as qtu
from src.qat import recalibrate
import mlflow

def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8용 NVIDIA TensorRT 최적화 방식 양자화-인지 학습(QAT)')
    parser.add_argument('--config', type=str, default='configs/config_qat.yaml', help='QAT 설정 파일 경로')
    parser.add_argument('--data', type=str, default=None, help='재학습 데이터 경로 덮어쓰기용(data.yaml)')
    parser.add_argument('--name', type=str, default=None, help='실험 이름 (예: run_id)')
    parser.add_argument('--weights', type=str, default=None, help='사전 학습 모델 가중치 경로 덮어쓰기용')
    return parser.parse_args()

def main():
    args = get_args()
    print("🚀 NVIDIA 기반 QAT 학습 파이프라인 시작 중...")
    
    # 1. 설정 로딩
    if not os.path.exists(args.config):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
        return
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"📄 설정 파일 로드 완료: {args.config}")

    # 커맨드라인 인자로 Config 매개변수 덮어쓰기
    if args.data:
        print(f"🔄 데이터 경로 덮어쓰기: {args.data}")
        # args.data가 data.yaml의 전체 경로라고 가정합니다.
        config['data_yaml_path'] = args.data 
    if args.name:
        print(f"🔄 실험명 덮어쓰기: {args.name}")
        config['exp_name'] = args.name
    if args.weights:
        print(f"🔄 사전학습 가중치 덮어쓰기: {args.weights}")
        config['qat']['pretrained_path'] = args.weights

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # 장치(Device) 설정
    device_str = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # 숫자형 '0' 입력을 'cuda:0' 형태로 변환합니다.
    if device_str.isdigit():
        device_str = f"cuda:{device_str}"
    device = torch.device(device_str)
    print(f"💻 연산 장치: {device}")
    
    # MLflow 설정 (통합 트래킹 구성)
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    mlflow.set_experiment("PCB_Retraining_Pipeline")
    
    # MLflow Run 시작
    with mlflow.start_run(run_name=config.get('exp_name', 'default')) as run:
        mlflow.log_params(config)
        
        # 2. QAT 환경 세팅 (NVIDIA 모듈 자동 주입)
        # 반드시 모델 로딩 "이전에" 설정해야 내부 레이어가 자동으로 양자화 포맷으로 교체됩니다.
        print("🔧 NVIDIA pytorch-quantization 기능 초기화 중...")
        qtu.initialize_quantization(config)

        # 3. 사전학습된 모델 로드
        pretrained_path = config['qat']['pretrained_path']
        if not os.path.isabs(pretrained_path):
            pretrained_path = os.path.join(project_root, pretrained_path)
        
        print(f"📦 사전학습 결과 가중치 로드 중: {pretrained_path}")
        if not os.path.exists(pretrained_path):
            print(f"❌ 가중치 파일을 찾을 수 없습니다: {pretrained_path}")
            sys.exit(1)

        # YOLO 객체 래퍼 로드
        yolo_wrapper = YOLO(pretrained_path) 
        model = yolo_wrapper.model
        
        # [중요] 강제로 내부 레이어를 깊게 탐색해 교체합니다. 
        # Ultralytics 특유의 구조가 자동 패치를 우회하는 현상을 막기 위함입니다.
        print("🔄 QAT 삽입을 위해 깊은 레이어 교체를 강제로 수행합니다...")
        qtu.replace_with_quantization_modules(model)
        
        model.to(device).float().train()

        # 민감한 레이어(Head, Attention 블록 등)의 양자화를 우회합니다.
        # 4. 캘리브레이션용 데이터 로더 준비 (깔끔한 이미지 사용)
        
        # data.yaml 경로 확인
        if 'data_yaml_path' in config:
             data_yaml = config['data_yaml_path']
        else:
             data_path = config.get('data_path', './PCB_DATASET')
             if not os.path.isabs(data_path):
                 data_path = os.path.join(project_root, data_path)
             data_yaml = os.path.join(data_path, 'data.yaml')

        imgsz = config.get('img_size', 640)
        batch_size = config.get('batch_size', 8)
        workers = config.get('workers', 4)

        print(f"📚 데이터 증강이 배제된 클린한 캘리브레이션 데이터 로드 시작 (imgsz={imgsz}, batch={batch_size})...")
        # 데이터 증강 기능이 빠진 전용 캘리브레이션 로더 함수 호출
        train_loader = qtu.get_calibration_dataloader(
            data_yaml_path=data_yaml,
            imgsz=imgsz,
            batch_size=batch_size,
            workers=workers
        )

        # 요약: 파인튜닝을 위한 객체 자체는 뒤에서 초기화합니댜.

        # [전체 양자화 전략] -> YOLOv11m 크기에서 작은 객체 성능 이슈 발생으로 하이브리드 방식으로 회귀.
        # 캘리브레이션이 성공하더라도 명시적인 Head 블록 양자화는 YOLOv11의 소형 객체 검출률을 치명적으로 떨어트립니다.
        if config['qat']['quantization'].get('skip_last_layers', True):
            print("🛡️  하이브리드 모드: 안정성을 위해 Head/의존성 레이어들의 양자화를 제외합니다...")
            qtu.disable_sensitive_layers_quantization(model)
        else:
            print("⚡ 전체 INT8 모드: 검출부 Head를 포함한 전체 양자화가 활성화되었습니다.")

        # 5. 통계치 수집 및 캘리브레이션
        # mAP를 결정하는 핵심인 스케일값(Scale) 추출 단계입니다.
        calib_batches = config['qat']['calibration'].get('num_batches', 64)
        calib_method = config['qat']['calibration'].get('method', 'mse')
        
        print(f"\n⚖️  캘리브레이션 적용 및 스케일 재보정 시작 ({calib_method})...")
        qtu.collect_calibration_stats(model, train_loader, num_batches=calib_batches, calib_method=calib_method, device=device)
        # [중요] 텐서 장치를 재지정합니다. load_calib_amax 단계에서 CPU 연산으로 넘어간 amax 값들을 복구시킵니다.
        model.to(device)
        print("✅ 캘리브레이션 종료. 스케일 값이 모델에 삽입되었습니다.")

        # 6. 커스텀 Trainer를 이용한 QAT Fine-Tuning (EMA/AMP Off 및 Warmup 우회 구조)
        print("\n▶️  Ultralytics Trainer 기반 QAT 파인튜닝 프로세스 시작...")
        
        # 커스텀 학습기 임포트
        from src.qat.trainer import QATDetectionTrainer
        
        # Trainer 매개변수 준비
        args_dict = {
            'model': pretrained_path, # 더미 경로 전달(이후 실제 모델 변수로 강제 교체)
            'data': data_yaml,
            'epochs': config['qat']['finetune']['epochs'],
            'imgsz': imgsz,
            'batch': batch_size,
            'workers': workers,
            'device': device_str,
            'optimizer': 'AdamW',
            'lr0': float(config['qat']['finetune']['lr0']),
            'lrf': 0.01,
            'weight_decay': config.get('weight_decay', 0.0005),
            'project': os.path.join(project_root, config.get('project', 'runs/qat')),
            'name': config.get('exp_name', 'default'),
            'exist_ok': True,
            'val': True,
            'plots': False,      # 디스크 용량 절감을 위해 plot 옵션 비활성화
            'verbose': False,    # [클린업] 클래스별 테이블 출력 끄기
            # Loss 가중치 설정
            'box': config.get('box', 7.5),
            'cls': config.get('cls', 0.5),
            'dfl': config.get('dfl', 1.5),
            # 증강 기법 (기본 config를 계승하거나 폴백)
            'hsv_h': config.get('hsv_h', 0.015),
            'hsv_s': config.get('hsv_s', 0.7),
            'hsv_v': config.get('hsv_v', 0.4),
        }
        
        # 트레이너 초기화
        trainer = QATDetectionTrainer(overrides=args_dict)
        
        # [중요] 기존 Trainer.model 객체를 QAT가 주입된 우리의 model 객체로 강제 덮어씌웁니다.
        # 내부 로딩 로직이나 다운로드 기능에 모델이 날아가는 상황을 방지합니다.
        trainer.model = model
        
        # 미세조정 시작
        train_results = trainer.train()
        
        print("🎉 양자화 인지 학습 사이클이 전부 완료되었습니다!")
        print(f"💾 체크포인트가 다음 위치에 담겼습니다: {trainer.save_dir}")

        # 7. 자동 EMA-FP32 추출 및 스케일 재보정 (원클릭)
        print("\n🔗 EMA 가중치와 하이브리드 형식으로 결합하는 모델 안정화(Recalibration) 수행...")
        best_pt_path = str(trainer.best)
        final_model_path = best_pt_path
        
        if os.path.exists(best_pt_path):
            try:
                # 모델 재보정
                hybrid_path = recalibrate.run_recalibration(args.config, best_pt_path, base_model_path=pretrained_path)
                final_model_path = hybrid_path
                print(f"✅ 완전한 QAT 파이프라인 마무리 완료. 하이브리드 안정화 모델 경로: {hybrid_path}")
                
            except Exception as e:
                print(f"❌ EMA 기반 모델 하이브리드화에 실패했습니다: {e}")
                print(f"⚠️  폴백 처리: 추출된 기본 best.pt 가중치를 사용하여 진행합니다: {best_pt_path}")
                fallback_path = best_pt_path.replace("best.pt", "best_qat_fallback.pt")
                import shutil
                shutil.copy(best_pt_path, fallback_path)
                final_model_path = fallback_path
        else:
            print(f"⚠️ {best_pt_path} 위치에 최고 모델이 존재하지 않습니다. 재보정을 통과합니다.")
            
        # 8. Test Set 최종 추론 및 모델 지표 등록
        print(f"\n🧪 {final_model_path} 경로의 최종 병합 파일을 통한 최종 테스트 셋 평가 시작...")
        try:
             # 평가를 위한 Base 가중치를 활용해 기본 뼈대 로드
             base_weights = config.get('qat', {}).get('pretrained_path', os.path.join(project_root, 'yolo11n.pt'))
             test_model = YOLO(base_weights)
             
             print("💉 테스트 추론을 위한 QAT 모듈들을 테스트 네트워크에 주입 중...")
             qtu.replace_with_quantization_modules(test_model.model)
             
             # 파인튜닝-EMA 결과가 합쳐진 가중치를 삽입합니다.
             print(f"📥 재학습 가중치 파일 로딩 시도 위치: {final_model_path}")
             ckpt = torch.load(final_model_path, map_location=device)
             if isinstance(ckpt, dict) and 'model' in ckpt:
                 test_model.model.load_state_dict(ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model'])
             else:
                 test_model.model.load_state_dict(ckpt)
             
             test_model.to(device)
             
             # 검증용 파티션 식별 (test or val)
             with open(data_yaml, 'r') as f:
                 dcfg = yaml.safe_load(f)
            
             split_to_eval = 'test' if 'test' in dcfg else 'val'
             print(f"   타겟 검증 데이터 분할 셋: {split_to_eval}")
             
             # 모델 성적표 추출 (MLflow 중복 충돌을 방지하여 plots/callback 생략)
             metrics = test_model.val(data=data_yaml, split=split_to_eval, batch=batch_size, device=device_str, plots=False)
             
             # 독립된 이름공간(`test/`)에 지표들을 개별 등록합니다.
             print(f"📊 최종 성능 기록(Metrics)을 MLflow 대시보드에 업로드 중...")
             mlflow.log_metric("test/mAP50", metrics.box.map50)
             mlflow.log_metric("test/mAP50_95", metrics.box.map)
             mlflow.log_metric("test/precision", metrics.box.mp)
             mlflow.log_metric("test/recall", metrics.box.mr)
             
             print(f"   테스트(Test) 결과 mAP50: {metrics.box.map50:.4f}")
             print(f"   테스트(Test) 결과 mAP50-95: {metrics.box.map:.4f}")
             
             # 완성된 재가중치 모델 파일도 훈련 서버 레지스트리에 함께 끼워넣습니다.
             mlflow.log_artifact(final_model_path, artifact_path="weights")
             
        except Exception as e:
            print(f"❌ 분석 테스트 실패. 에러 원인 통계: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
