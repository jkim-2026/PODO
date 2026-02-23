import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import sys
import os

# 스크립트 실행 위치에 관계없이 src 모듈을 임포트하기 위해 프로젝트 루트를 경로에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.qat import utils as qtu
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(description='안정적인 NVIDIA QAT 모델 ONNX 추출기')
    parser.add_argument('--weights', type=str, required=True, help='학습이 완료된 best_qat.pt 파일 경로')
    parser.add_argument('--base-weights', type=str, required=True, help='구조(Structure) 유지를 위한 원본 FP32 best.pt 파일 경로')
    parser.add_argument('--imgsz', type=int, default=640, help='입력 이미지 크기 (Image size)')
    parser.add_argument('--output', type=str, default=None, help='출력될 ONNX 파일 경로')
    parser.add_argument('--full-int8', action='store_true', help='전체(Head 포함) 양자화 모드 활성화 (하이브리드 전략 비활성화)')
    parser.add_argument('--asymmetric', action='store_true', help='비대칭 양자화 사용 (주의: TensorRT 불일치 발생 가능)')
    parser.add_argument('--force-raw', action='store_true', help='EMA 가중치 대신 Raw(모델 원본) 가중치 강제 사용')
    return parser.parse_args()

def main():
    args = get_args()
    print("🚀 NVIDIA QAT ONNX 변환 시작 (안정화 버전)...")
    
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        print("❌ pytorch-quantization 라이브러리를 찾을 수 없습니다. 설치 후 진행하세요.")
        sys.exit(1)

    # 1. 양자화(QAT) 컨텍스트(Context) 초기화
    print("🔧 양자화 모듈들을 초기화하고 교체 준비 중...")
    
    # 인자로 받은 값에 따라 양자화 환경(Config) 구성
    config = {'qat': {'quantization': {'weight_per_channel': True, 'num_bits': 8}, 'calibration': {'method': 'mse'}}}
    
    if args.asymmetric:
        print("⚠️  경고: 비대칭 양자화(Asymmetric Quantization)가 켜져 있습니다. TensorRT 변환 시 문제가 발생할 수 있습니다.")
        config['qat']['quantization']['symmetric'] = False
    else:
        print("✅ 대칭 양자화(Symmetric Quantization)를 통한 엔진 보호 활성화 완료.")
        config['qat']['quantization']['symmetric'] = True

    qtu.initialize_quantization(config)

    # 2. Ultralytics를 이용해 베이스 모델 구조 생성 (FP32 기준)
    print(f"🏗️  다음 경로에서 베이스 모델 구조를 읽어오는 중: {args.base_weights}...")
    model_wrapper = YOLO(args.base_weights)
    model = model_wrapper.model
    
    # 3. 모델 안의 기존 레이어를 양자화 레이어로 재귀적 교체(Deep Recursive Replacement)
    qtu.replace_with_quantization_modules(model)
    
    # [하이브리드 모드 체크]
    if not args.full_int8:
        print("🛡️  하이브리드 모드: 안정성을 위해 민감한 레이어(Head 등)의 양자화를 비활성화했습니다...")
        qtu.disable_sensitive_layers_quantization(model)
    else:
        print("⚠️  전체 INT8 모드: 검출부 Head가 양자화 대상에 포함되었습니다. 정확도(mAP) 하락에 주의하세요.")
    
    # 4. 저장된 QAT 상태 사전(State Dict) 로드
    print(f"📂 다음 경로에서 QAT 가중치를 디코딩 중: {args.weights}...")
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # 가중치 파일 구조에서 가장 최적의 모델(State Dict) 부분 추출 (안정성 보장)
    state_dict = None
    if isinstance(checkpoint, dict):
        if args.force_raw and 'model' in checkpoint:
            print("📦 강제성 확보: EMA 가중치가 있더라도 원본 모델('model' 키) 가중치로 덮어씁니다.")
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint and checkpoint['ema'] is not None:
            print("✨ EMA 가중치를 감지했습니다. 정확도 향상을 위해 EMA 값을 채택합니다.")
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            print("📦 원본 모델(model) 가중치를 감지했습니다.")
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 추출된 값이 상태 사전(dict)이 아니라 모델 객체 그 자체일 경우
    if hasattr(state_dict, 'state_dict'):
        print("🔄 모델 객체를 내부의 state_dict로 변환 중...")
        state_dict = state_dict.state_dict()
    
    # [핵심 수정] 무작정 접두사를 제거하지 마십시오.
    # Ultralytics 기반 모델은 'model.0...' 같은 키워드를 기대하므로 이 구조를 파괴해선 안 됩니다.
    # 모델 병렬 처리 구조인 'module.' 래핑 형태(DDP)만 필터링합니다.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 변환된 가중치를 그대로 모델에 매핑해 봅니다.
    try:
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 가중치 로드 성공. (누락: {len(msg.missing_keys)}개, 불일치: {len(msg.unexpected_keys)}개)")
    except RuntimeError:
        # Fallback 조치: 타겟이 Sequential 컨테이너일 경우를 대비해 'model.' 접두사 강제 제거 시도
        print("⚠️ 직접 로딩 실패. 방어 코드를 통해 'model.' 접두사 소거 메커니즘을 가동합니다...")
        fallback_state_dict = OrderedDict()
        for k, v in new_state_dict.items():
             if k.startswith('model.'):
                 fallback_state_dict[k[6:]] = v
             else:
                 fallback_state_dict[k] = v
        msg = model.load_state_dict(fallback_state_dict, strict=False)
        print(f"⚠️ 임시 조치 완료. (누락: {len(msg.missing_keys)}개, 불일치: {len(msg.unexpected_keys)}개)")

    model.eval()
    model.float() # [버그 픽스] 반드시 FP32 상태여야 양자화 ONNX 변환이 정상적으로 수행됩니다 (FP16 회피).
    if torch.cuda.is_available(): model.cuda()

    # 5. ONNX 특화 모드로 텐서 연산 조율
    print("⚙️  ONNX 추출을 위해 출력 레이어(Detect Layer) 특성(Property) 조정 중...")
    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
            m.format = 'onnx'

    # 6. 활성화(Amax) 상태 체크 및 최종 내보내기 활성화
    qtu.prepare_model_for_export(model)

    # 7. ONNX Export 실행 (PyTorch 내장 기능 활용)
    if not args.output:
        args.output = args.weights.replace('.pt', '.onnx')
    
    # 안정적인 경로 해석 (Robust Path Handling)
    from pathlib import Path
    out_path = Path(args.output)
    if out_path.suffix != '.onnx':
        if out_path.exists() and out_path.is_dir():
             out_path = out_path / Path(args.weights).stem
        elif args.output.endswith('/'):
             out_path = out_path / Path(args.weights).stem
             
        out_path = out_path.with_suffix('.onnx')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(out_path)
    
    print(f"📦 {args.output} 위치로 모델 추출 중... (ONNX Opset Version 13)")
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz).to(next(model.parameters()).device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            opset_version=13,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
        )
        print("✅ ONNX 파일 생성이 성공했습니다.")

        # 8. YOLO 모델용 메타데이터 병합 주입 -> (필수 사항: TensorRT / Ultralytics 런타임용)
        print("💉 ONNX 파일 단편을 열람하여 필수 YOLO 메타데이터(클래스명 등)를 주입합니다...")
        import onnx
        onnx_model = onnx.load(args.output)
        
        # Ultralytics를 위해 요구되는 메타데이터 구성
        meta_dict = {
            'names': eval(str(model.names)) if hasattr(model, 'names') else {i: f'class_{i}' for i in range(100)},
            'stride': int(max(model.stride.cpu().numpy())) if hasattr(model, 'stride') else 32,
            'task': 'detect',
            'batch': 1,
            'imgsz': [args.imgsz, args.imgsz]
        }
        
        # 키/값 형태로 파싱하여 모델 속성으로 추가
        for k, v in meta_dict.items():
            meta = onnx_model.metadata_props.add()
            meta.key = k
            meta.value = str(v)
            
        onnx.save(onnx_model, args.output)
        print("✅ 메타데이터 주입 완료.")
        
        # 무결성 검증 수행
        print("\n🔍 양자화 노드(Q/DQ)의 무결성을 보일 수 있도록 ONNX Graph 구조를 검사합니다...")
        try:
            q_count = 0
            for node in onnx_model.graph.node:
                 if node.op_type == 'QuantizeLinear': q_count += 1
            
            if q_count > 0:
                print(f"🎉 검증 통과: 총합 {q_count}개의 QuantizeLinear(INT8) 노드가 발견되었습니다!")
            else:
                print("⚠️  경고: 표준 13 Opset 구조 내에서 Q/DQ 블록을 하나도 찾지 못했습니다.")
                print("💡 pytorch-quantization 모듈이 정상적으로 가중치를 양자화했는지 다시 점검하세요.")
        except ImportError:
            print("ℹ️  onnx 파이썬 패키지 내부 오류로, Graph 검증 절차를 생략합니다.")

    except Exception as e:
        print(f"❌ 데이터 추출 작업 실패 (Export Failed): {e}")
        import sys
        sys.exit(1)
    finally:
        # 공유 메모리 또는 FB(Facebook) 타입 양자화 텐서를 사용 해제 처리
        quant_nn.TensorQuantizer.use_fb_fake_quant = False

if __name__ == "__main__":
    main()
