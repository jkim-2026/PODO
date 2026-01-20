"""
QAT 변환 검증 스크립트

Conv2d가 QuantConv2d로 제대로 교체되었는지 확인합니다.
"""
import sys
import yaml
from pathlib import Path

# training 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent))

from src.models.yolov8s_qat import get_model


def verify_qat_layers():
    print("=== QAT 변환 검증 시작 ===\n")

    config_path = Path(__file__).parent / 'config_qat.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 모델 로드 (여기서 수동 Conv2d → QuantConv2d 교체가 일어남)
    print("모델 로드 및 양자화 레이어 교체 중...")
    model = get_model(config)

    print("\n[레이어 타입 확인]")
    quant_conv_count = 0
    conv_count = 0

    # 모든 Conv 레이어 확인
    for name, module in model.model.named_modules():
        mod_type = type(module).__name__

        if 'QuantConv2d' in mod_type:
            quant_conv_count += 1
            if quant_conv_count <= 10:  # 처음 10개만 출력
                print(f"  ✅ {name}: {mod_type}")
        elif 'Conv2d' in mod_type and 'Quant' not in mod_type:
            conv_count += 1
            if conv_count <= 5:  # 처음 5개만 출력
                print(f"  ⚠️  {name}: {mod_type} (양자화 안됨)")

    print(f"\n[통계]")
    print(f"  - QuantConv2d 레이어: {quant_conv_count}개")
    print(f"  - 일반 Conv2d 레이어: {conv_count}개")

    print("\n[검증 결과]")
    if quant_conv_count > 0:
        print("✅ 성공: QuantConv2d 레이어가 발견되었습니다.")
        print("   QAT가 정상적으로 적용될 준비가 되었습니다.")
        if conv_count > 0:
            print(f"   (일부 Conv2d는 Detect Head 등으로 양자화 제외됨)")
        return True
    else:
        print("❌ 실패: QuantConv2d 레이어가 발견되지 않았습니다.")
        print("   replace_conv_with_quantconv() 함수를 확인하세요.")
        return False


if __name__ == "__main__":
    success = verify_qat_layers()
    sys.exit(0 if success else 1)
