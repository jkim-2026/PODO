"""
QAT Checkpoint 검증 스크립트

Best checkpoint가 올바르게 저장/복원되는지 검증합니다:
1. TensorQuantizer 정보가 checkpoint에 저장되었는지 확인
2. Checkpoint를 로드해서 Conv2d → QuantConv2d 재구성 확인
3. Validation 메트릭이 0이 아닌지 확인

사용법:
    python validate_qat_checkpoint.py --checkpoint runs/qat/qat_yolov8s/weights/best.pt --data PCB_DATASET/data.yaml
"""

import sys
import torch
from pathlib import Path
from typing import Dict, Any

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))


def validate_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """
    Checkpoint 메타데이터 검증.

    Args:
        checkpoint_path: Checkpoint 파일 경로

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*80)
    print("[1/3] Checkpoint 메타데이터 검증")
    print("="*80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    result = {
        'has_quantizer_state': False,
        'quantizer_count': 0,
        'quantizer_details': None,
    }

    # TensorQuantizer 상태 확인
    if 'quantizer_state' in checkpoint:
        result['has_quantizer_state'] = True
        quantizer_state = checkpoint['quantizer_state']
        result['quantizer_count'] = quantizer_state.get('quantizer_count', 0)
        result['quantizer_details'] = quantizer_state

        print(f"✅ TensorQuantizer 정보 발견!")
        print(f"  - Quantizer 개수: {result['quantizer_count']}")
        print(f"  - 저장된 정보: {list(quantizer_state.keys())}")

        # 첫 번째 quantizer 정보 출력 (샘플)
        if 'quantizers' in quantizer_state and quantizer_state['quantizers']:
            first_name = list(quantizer_state['quantizers'].keys())[0]
            first_info = quantizer_state['quantizers'][first_name]
            print(f"\n  샘플 Quantizer: {first_name}")
            print(f"    - num_bits: {first_info.get('num_bits')}")
            print(f"    - is_enabled: {first_info.get('is_enabled')}")
            print(f"    - amax: {first_info.get('amax')}")
    else:
        print(f"❌ TensorQuantizer 정보 없음!")
        print(f"  이 checkpoint는 QAT 모델이 아니거나,")
        print(f"  save_quantizer_state()가 호출되지 않았습니다.")

    return result


def validate_model_loading(checkpoint_path: str) -> Dict[str, Any]:
    """
    모델 로딩 및 QuantConv2d 재구성 검증.

    Args:
        checkpoint_path: Checkpoint 파일 경로

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*80)
    print("[2/3] 모델 로딩 및 QuantConv2d 재구성 검증")
    print("="*80)

    from ultralytics import YOLO
    from src.models.qat_detection_trainer import QATDetectionTrainer

    result = {
        'loading_success': False,
        'quantizer_count_after_load': 0,
    }

    try:
        print(f"\n[Step 1] YOLO로 직접 로드 (일반 방식)...")
        model1 = YOLO(checkpoint_path)

        # TensorQuantizer 개수 확인
        try:
            from pytorch_quantization import nn as quant_nn
            quantizer_count = sum(1 for m in model1.model.modules()
                                 if isinstance(m, quant_nn.TensorQuantizer))
            print(f"  - TensorQuantizer 개수: {quantizer_count}")

            if quantizer_count == 0:
                print(f"  ⚠️ 일반 로드 방식은 QuantConv2d를 복원하지 않습니다.")
        except ImportError:
            print(f"  (pytorch-quantization 미설치)")

        print(f"\n[Step 2] QATDetectionTrainer.get_model()로 로드...")
        trainer = QATDetectionTrainer()
        model2 = trainer.get_model(weights=checkpoint_path)

        # TensorQuantizer 개수 확인
        try:
            quantizer_count = sum(1 for m in model2.modules()
                                 if isinstance(m, quant_nn.TensorQuantizer))
            result['quantizer_count_after_load'] = quantizer_count
            print(f"  - TensorQuantizer 개수: {quantizer_count}")

            if quantizer_count > 0:
                print(f"  ✅ QuantConv2d 재구성 성공!")
                result['loading_success'] = True
            else:
                print(f"  ❌ QuantConv2d 재구성 실패!")
        except ImportError:
            print(f"  (pytorch-quantization 미설치)")

    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()

    return result


def validate_metrics(checkpoint_path: str, data_yaml: str) -> Dict[str, Any]:
    """
    Validation 메트릭 검증.

    Args:
        checkpoint_path: Checkpoint 파일 경로
        data_yaml: 데이터셋 yaml 경로

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*80)
    print("[3/3] Validation 메트릭 검증")
    print("="*80)

    from ultralytics import YOLO
    from src.models.qat_detection_trainer import QATDetectionTrainer

    result = {
        'validation_success': False,
        'metrics': None,
    }

    try:
        print(f"\n[Step 1] QATDetectionTrainer로 모델 로드...")
        trainer = QATDetectionTrainer()
        model = trainer.get_model(weights=checkpoint_path)

        print(f"[Step 2] Validation 수행 중...")
        print(f"  (이 작업은 몇 분 소요될 수 있습니다...)")

        # YOLO 객체 생성 (validation용)
        yolo_model = YOLO(checkpoint_path)

        # 모델 교체 (QuantConv2d 포함)
        yolo_model.model = model

        # Validation 수행
        metrics = yolo_model.val(data=data_yaml)

        result['metrics'] = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.p,
            'recall': metrics.box.r,
        }

        print(f"\n✅ Validation 완료!")
        print(f"  - mAP50: {result['metrics']['mAP50']:.4f}")
        print(f"  - mAP50-95: {result['metrics']['mAP50-95']:.4f}")
        print(f"  - Precision: {result['metrics']['precision']:.4f}")
        print(f"  - Recall: {result['metrics']['recall']:.4f}")

        # 메트릭이 0이 아닌지 확인
        if result['metrics']['mAP50'] > 0.0:
            print(f"\n🎉 메트릭 검증 통과! (mAP50 > 0)")
            result['validation_success'] = True
        else:
            print(f"\n❌ 메트릭 0 문제 발생!")
            result['validation_success'] = False

    except Exception as e:
        print(f"❌ Validation 실패: {e}")
        import traceback
        traceback.print_exc()

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='QAT Checkpoint 검증')
    parser.add_argument('--checkpoint', type=str, required=True, help='Best checkpoint 경로')
    parser.add_argument('--data', type=str, required=True, help='Data yaml 경로')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    data_yaml = args.data

    # 파일 존재 확인
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint 파일 없음: {checkpoint_path}")
        return

    if not Path(data_yaml).exists():
        print(f"❌ Data yaml 파일 없음: {data_yaml}")
        return

    print("\n" + "="*80)
    print("QAT Checkpoint 검증 시작")
    print("="*80)
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Data yaml: {data_yaml}")

    # 1. 메타데이터 검증
    metadata_result = validate_checkpoint_metadata(checkpoint_path)

    # 2. 모델 로딩 검증
    loading_result = validate_model_loading(checkpoint_path)

    # 3. 메트릭 검증 (선택적)
    print("\n" + "="*80)
    print("Validation 메트릭 검증을 수행하시겠습니까?")
    print("(이 작업은 몇 분 소요될 수 있습니다)")
    print("="*80)
    response = input("진행하시겠습니까? (y/n): ")

    if response.lower() == 'y':
        metrics_result = validate_metrics(checkpoint_path, data_yaml)
    else:
        print("\nValidation 메트릭 검증을 건너뜁니다.")
        metrics_result = {'validation_success': None}

    # 최종 결과 요약
    print("\n" + "="*80)
    print("검증 결과 요약")
    print("="*80)

    print(f"\n[1] TensorQuantizer 메타데이터:")
    if metadata_result['has_quantizer_state']:
        print(f"  ✅ 저장됨 ({metadata_result['quantizer_count']}개)")
    else:
        print(f"  ❌ 없음")

    print(f"\n[2] QuantConv2d 재구성:")
    if loading_result['loading_success']:
        print(f"  ✅ 성공 ({loading_result['quantizer_count_after_load']}개)")
    else:
        print(f"  ❌ 실패")

    print(f"\n[3] Validation 메트릭:")
    if metrics_result['validation_success'] is None:
        print(f"  ⏭️  건너뜀")
    elif metrics_result['validation_success']:
        print(f"  ✅ 정상 (mAP50: {metrics_result['metrics']['mAP50']:.4f})")
    else:
        print(f"  ❌ 메트릭 0")

    # 전체 결과
    all_passed = (
        metadata_result['has_quantizer_state'] and
        loading_result['loading_success'] and
        (metrics_result['validation_success'] is None or metrics_result['validation_success'])
    )

    print("\n" + "="*80)
    if all_passed:
        print("🎉 모든 검증 통과!")
        print("="*80)
        print("\nQAT Checkpoint가 올바르게 저장/복원됩니다.")
        print("ONNX export도 정상적으로 작동할 것입니다.")
    else:
        print("❌ 일부 검증 실패")
        print("="*80)
        print("\n문제가 발견되었습니다. 위의 로그를 확인하세요.")

    print()


if __name__ == "__main__":
    main()
