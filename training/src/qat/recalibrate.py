import argparse
import os
import sys
import yaml
import torch
import copy
from ultralytics import YOLO
from . import utils as qtu
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='QAT를 위한 하이브리드 EMA(지수 이동 평균) 재보정(Re-calibration)')
    parser.add_argument('--config', type=str, required=True, help='config_qat.yaml 파일 경로')
    parser.add_argument('--weights', type=str, required=True, help='학습된 대상 체크포인트(best_qat.pt) 경로')
    parser.add_argument('--output', type=str, default=None, help='출력될 하이브리드 모델 저장 경로')
    return parser.parse_args()

def run_recalibration(config_path, weights_path, output_path=None, base_model_path=None):
    """
    프로그래밍 방식으로 EMA 재보정 프로세스를 실행합니다.
    
    Args:
        config_path (str): QAT 설정 YAML 파일 경로.
        weights_path (str): 학습이 완료된 체크포인트 파일 경로(best.pt).
        output_path (str, optional): 사용자 지정 출력 경로. 기본값은 원본에 _hybrid 가 붙습니다.
        base_model_path (str, optional): 아키텍처 생성을 위한 사전 학습 모델 로드 경로.
    
    Returns:
        str: 저장된 하이브리드 모델의 최종 경로.
    """
    print(f"\n🚀 하이브리드 EMA 재보정 시작...")
    print(f"   설정 파일: {config_path}")
    print(f"   가중치 파일: {weights_path}")

    # 1. 환경 설정(Config) 불러오기
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 양자화(QAT) 컨텍스트(Context) 초기화
    print("🔧 양자화 모듈들을 초기화 중...")
    qtu.initialize_quantization(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', 'cuda') != 'cpu' else 'cpu')
    print(f"💻 할당된 디바이스: {device}")

    # 3. 모델의 구조(Skeleton) 구축
    if not base_model_path:
        base_model_path = config['qat']['pretrained_path']
        
    print(f"🏗️  {base_model_path} 파라미터를 기반으로 모델 뼈대 생성 중...")
    model_wrapper = YOLO(base_model_path)
    model = model_wrapper.model
    
    # 4. QAT 전용 모듈들의 주입 및 예외 계층 비활성화
    qtu.replace_with_quantization_modules(model)
    qtu.disable_sensitive_layers_quantization(model)
    model.to(device)

    # 5. EMA 방식의 가중치 불러오기
    print(f"📂 {weights_path} 위치에서 체크포인트를 불러옵니다...")
    ckpt = torch.load(weights_path, map_location='cpu')
    
    ema_weights = None
    if isinstance(ckpt, dict):
        if 'ema' in ckpt and ckpt['ema'] is not None:
            print("✨ 체크포인트에서 'ema' 키 발견! 품질 향상을 위해 EMA 가중치를 채택합니다.")
            ema_obj = ckpt['ema']
            ema_weights = ema_obj
        else:
            print("⚠️  'ema' 키를 찾을 수 없으므로, 기본 'model' 키를 탐색합니다...")
            ema_weights = ckpt.get('model', ckpt)
    else:
        ema_weights = ckpt

    # 추출된 값이 상태 사전(dict)이 아니라 모델 객체 그 자체일 경우에 대비한 추출 로직
    if hasattr(ema_weights, 'model') and isinstance(ema_weights.model, torch.nn.Module):
         # 모델이 Ultralytics 래퍼로 감싸져 있는 경우
         print("📦 Ultralytics 래퍼 구조 감지됨. 내부 실제 구조체(.model)를 분리합니다...")
         ema_weights = ema_weights.model

    if hasattr(ema_weights, 'state_dict'):
        print("🔄 추출된 EMA 모델 객체를 저장 가능 규격(state_dict)으로 변환 중...")
        try:
            ema_weights = ema_weights.state_dict()
        except Exception as e:
             print(f"⚠️  state_dict() 메서드를 호출할 수 없습니다: {e}")

    if ema_weights is None:
        print("❌ 심각한 에러: 가중치 추출에 완전히 실패했습니다.")
        return None
        
    # 안정적인 가중치 로딩 (접두사 처리 규칙)
    new_state_dict = OrderedDict()
    for k, v in ema_weights.items():
        # 맹목적으로 'model.' 접두사를 자르지 마십시오.
        # 타겟 모델 구조를 신뢰하고 있는 그대로 복사합니다. DetectionModel은 주로 'model.0...'을 기대합니다.
        new_state_dict[k] = v

    # 변환된 가중치 그대로 매핑 시도
    try:
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 가중치 로드 성공. (누락: {len(msg.missing_keys)}개, 불일치: {len(msg.unexpected_keys)}개)")
    except RuntimeError:
        # Fallback 처리: 'model.' 소거
        print("⚠️ 직접 로딩 실패. 방어 코드를 통해 'model.' 접두사 소거 시도...")
        stripped = OrderedDict()
        for k, v in new_state_dict.items():
            if k.startswith('model.'):
                stripped[k[6:]] = v
            else:
                stripped[k] = v
        msg = model.load_state_dict(stripped, strict=False)
        print(f"✅ 임시 조치 성공(Fallback loaded). (누락: {len(msg.missing_keys)}개, 불일치: {len(msg.unexpected_keys)}개)")
    
    # 6. 재보정(Calibration)을 위한 데이터셋 준비
    print("📚 보정용 데이터셋을 설정하고 있습니다...")
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from types import SimpleNamespace
    from ultralytics.utils import DEFAULT_CFG_DICT

    # 데이터 설정(yaml) 불러오기
    data_yaml_path = config['data_path'] + '/data.yaml'
    if not os.path.exists(data_yaml_path):
        # 환경변수에 없으면 절대경로 로직으로 복귀 시도
        data_yaml_path = 'PCB_DATASET/data.yaml' 
        
    data_cfg = check_det_dataset(data_yaml_path)
    
    dataset_path = data_cfg['train']
    if isinstance(dataset_path, list): dataset_path = dataset_path[0] 
    
    # 상대 경로인 경우 올바르게 해석
    if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
         # Try relative to config['data_path']
         dataset_path = os.path.join(config['data_path'], dataset_path)

    batch_size = config.get('batch_size', 8)
    img_size = config.get('img_size', 640)
    
    # 임시 빌드를 위한 더미 구성(Dummy CFG) 생성
    cfg_dict = DEFAULT_CFG_DICT.copy()
    cfg_dict.update({
        'imgsz': img_size,
        'rect': False,
        'stride': 32,
        'batch': batch_size,
        'task': 'detect',
        'data': data_cfg,
        'mode': 'train',
        'single_cls': False,
    })
    dummy_cfg = SimpleNamespace(**cfg_dict)

    dataset = build_yolo_dataset(
        cfg=dummy_cfg, 
        img_path=dataset_path, 
        batch=batch_size, 
        data=data_cfg, 
        mode='train', 
        rect=False, 
        stride=32
    )
    
    train_loader = build_dataloader(dataset, batch=batch_size, workers=4, shuffle=True, rank=-1)

    # 7. 재보정(Re-Calibration) 실행
    calib_method = config['qat']['calibration']['method']
    num_batches = config['qat']['calibration']['num_batches']
    
    print(f"⚖️  추출된 EMA 가중치로 보정 절차 진행 중... (분석기법: {calib_method}, 허용 배치 수: {num_batches})")
    qtu.collect_calibration_stats(model, train_loader, num_batches=num_batches, calib_method=calib_method, device=device)
    
    # 8. 최종 하이브리드 모델 저장 (Save)
    if output_path is None:
        p_weights = Path(weights_path)
        # 접미사 이름 교정 및 파일경로 구조 세팅
        stem = p_weights.stem
        if 'qat' not in stem: stem += '_qat'
        output_path = str(p_weights.parent / f"{stem}_hybrid.pt")
        
    print(f"💾 하이브리드 QAT 보정 모델을 저장하고 있습니다: {output_path}")
    
    # 중요: 차후 ONNX 추출 과정에서의 크래시(Crash) 방지를 위해 무조건 FP32 타입으로 되돌려 놓습니다.
    model.cpu().float()
    
    save_ckpt = {
        'model': model,
        'ema': None, 
        'optimizer': None,
        'info': 'Hybrid EMA+Calib Model',
        'date': str(os.path.getmtime(weights_path))
    }
    torch.save(save_ckpt, output_path)
    print("🎉 재보정 작업 완료! (Recalibration Done)")
    
    return output_path

def main():
    args = get_args()
    run_recalibration(args.config, args.weights, args.output)

if __name__ == "__main__":
    main()
