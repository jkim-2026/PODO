"""
QAT Best Checkpoint → ONNX Export 전용 스크립트

학습이 완료된 QAT checkpoint를 ONNX로 export합니다.
참조 리포지토리 방식(NVIDIA TensorRT)을 우선 시도합니다.

사용법:
    python export_onnx_only.py --checkpoint runs/qat/qat_yolov8s_medium/weights/best.pt --config config_qat.yaml
"""

import sys
import torch
import yaml
from pathlib import Path

# Add training directory to path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='QAT Checkpoint ONNX Export')
    parser.add_argument('--checkpoint', type=str, required=True, help='Best checkpoint 경로')
    parser.add_argument('--config', type=str, required=True, help='Config yaml 경로')
    parser.add_argument('--output', type=str, default=None, help='ONNX 출력 경로 (기본: checkpoint와 같은 디렉토리)')
    parser.add_argument('--img-size', type=int, default=640, help='입력 이미지 크기')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)

    # 파일 존재 확인
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint 파일 없음: {checkpoint_path}")
        return

    if not config_path.exists():
        print(f"❌ Config 파일 없음: {config_path}")
        return

    # Config 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 출력 경로 설정
    if args.output:
        onnx_path = Path(args.output)
    else:
        onnx_path = checkpoint_path.parent / f"{checkpoint_path.stem}_qat.onnx"

    print("\n" + "="*80)
    print("QAT Checkpoint → ONNX Export")
    print("="*80)
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Config: {config_path}")
    print(f"  - Output: {onnx_path}")
    print(f"  - Image size: {args.img_size}")
    print(f"  - Device: {args.device}")

    # ========================================================================
    # 1. QAT 모델 로드
    # ========================================================================
    print("\n[1/2] QAT 모델 로드 중...")
    print("-"*80)

    from src.models.qat_detection_trainer import QATDetectionTrainer
    from pytorch_quantization import nn as quant_nn

    try:
        # Trainer 생성
        trainer = QATDetectionTrainer()

        # 모델 로드 (QuantConv2d 재구성 포함)
        print(f"[QAT] Checkpoint 로드: {checkpoint_path}")
        model = trainer.get_model(weights=str(checkpoint_path))

        # TensorQuantizer 개수 확인
        quantizer_count = sum(1 for m in model.modules()
                             if isinstance(m, quant_nn.TensorQuantizer))
        print(f"[QAT] ✅ 모델 로드 완료")
        print(f"  - TensorQuantizer 개수: {quantizer_count}")

        if quantizer_count == 0:
            print(f"[QAT] ⚠️ 경고: TensorQuantizer가 없습니다!")
            print(f"  ONNX export에 Q/DQ 노드가 포함되지 않을 수 있습니다.")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # 2. ONNX Export
    # ========================================================================
    print("\n[2/2] ONNX Export 시작...")
    print("-"*80)

    try:
        from src.quantization import export_qat_to_onnx

        # Export
        export_qat_to_onnx(
            model=model,
            output_path=str(onnx_path),
            config=config,
            img_size=args.img_size,
            device=args.device,
        )

        # Q/DQ 노드 확인
        if onnx_path.exists():
            file_size = onnx_path.stat().st_size / 1024 / 1024
            print(f"\n[QAT] ✅ ONNX export 성공!")
            print(f"  - 경로: {onnx_path}")
            print(f"  - 크기: {file_size:.1f} MB")

            # Q/DQ 노드 확인
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                qdq_count = {}
                for node in onnx_model.graph.node:
                    if node.op_type in ['QuantizeLinear', 'DequantizeLinear']:
                        qdq_count[node.op_type] = qdq_count.get(node.op_type, 0) + 1

                if qdq_count:
                    print(f"\n[QAT] 🎉 Q/DQ 노드 발견!")
                    for op, count in qdq_count.items():
                        print(f"  - {op}: {count}개")
                    print(f"\n[QAT] ✅ TensorRT INT8 변환 준비 완료!")
                    print(f"  엣지에서 calibration 없이 바로 TensorRT 엔진 생성 가능합니다.")
                else:
                    print(f"\n[QAT] ⚠️ Q/DQ 노드 없음")
                    print(f"  ONNX export 과정에서 Q/DQ 노드가 손실되었습니다.")
                    print(f"  엣지에서 PTQ calibration이 필요할 수 있습니다.")
            except Exception as e:
                print(f"[QAT] Q/DQ 노드 확인 실패: {e}")
        else:
            print(f"\n[QAT] ❌ ONNX 파일이 생성되지 않았습니다.")

    except Exception as e:
        print(f"\n[QAT] ❌ ONNX export 실패: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("완료")
    print("="*80)


if __name__ == "__main__":
    main()
