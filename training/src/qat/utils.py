"""
NVIDIA pytorch-quantization 툴킷을 활용하는 QAT 전용 유틸리티 모듈입니다.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from tqdm import tqdm

# 런타임에 이뤄지는 몽키 패치(Monkey-patching) 이후 isinstance 검사 실패를 방지하기 위해 네이티브 모듈을 직접 임포트합니다.
from torch.nn import Conv2d as NativeConv2d
from torch.nn import Linear as NativeLinear

def initialize_quantization(config: Dict[str, Any]) -> None:
    """pytorch-quantization 라이브러리를 초기화하고 기본 디스크립터(Descriptor)를 설정합니다."""
    try:
        from pytorch_quantization import quant_modules
        from pytorch_quantization.tensor_quant import QuantDescriptor
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        raise ImportError("pytorch-quantization 라이브러리가 설치되지 않았습니다.")

    # pytorch-quantization에서 발생하는 불필요한 로그(verbose logs)를 억제합니다.
    import logging
    logging.getLogger("pytorch_quantization.nn.modules.tensor_quantizer").setLevel(logging.ERROR)
    
    # 가능한 경우 absl 로깅 또한 억제합니다 (Google 기반 라이브러리에서 흔히 발생함).
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        pass

    qat_config = config.get('qat', {})
    quant_config = qat_config.get('quantization', {})
    calibration_config = qat_config.get('calibration', {})
    
    quant_modules.initialize()

    num_bits = quant_config.get('num_bits', 8)
    weight_per_channel = quant_config.get('weight_per_channel', True)
    calib_method = calibration_config.get('method', 'mse')

    symmetric = quant_config.get('symmetric', True) # 기본값 True (TensorRT와 호환성 유지)

    if calib_method in ['mse', 'entropy', 'histogram']:
        input_desc = QuantDescriptor(num_bits=num_bits, calib_method='histogram')
    else:
        input_desc = QuantDescriptor(num_bits=num_bits, calib_method='max')

    if weight_per_channel:
        if symmetric:
            # [핵심] TensorRT는 대칭 양자화(Symmetric Quantization, Zero Point가 0)를 권장/요구합니다.
            weight_desc = QuantDescriptor(num_bits=num_bits, axis=(0,), unsigned=False, narrow_range=True)
        else:
            # 비대칭 양자화 (PyTorch 기본 설정) - TensorRT 변환 시 불일치(Mismatch) 위험이 존재합니다.
            weight_desc = QuantDescriptor(num_bits=num_bits, axis=(0,))
    else:
        if symmetric:
            weight_desc = QuantDescriptor(num_bits=num_bits, unsigned=False, narrow_range=True)
        else:
            weight_desc = QuantDescriptor(num_bits=num_bits)

    quant_nn.QuantConv2d.set_default_quant_desc_input(input_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(weight_desc)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

    print(f"✅ QAT 초기화 완료: Bits={num_bits}, 채널별 가중치(WeightPerChannel)={weight_per_channel}, 보정기법(Calib)={calib_method}")
    print(f"   대칭 가중치(Symmetric Weights) 사용: {symmetric} (TensorRT 호환성: {'강력함(YES)' if symmetric else '보장안됨(NO)'})")

def replace_with_quantization_modules(model: nn.Module) -> None:
    """
    기존 레이어(nn.Conv2d, nn.Linear)를 양자화 레이어(QuantConv2d, QuantLinear)로 직접 교체합니다.
    nn.Sequential, nn.ModuleList 및 중첩 구조까지 재귀적으로 처리하여 누락을 방지합니다.
    몽키 패치 이후에도 정확한 레이어 식별을 위해 NativeConv2d를 사용합니다.
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    print("🔄 수동으로 모델 레이어들을 양자화 모듈로 교체하고 있습니다 (심층 재귀 탐색)...")
    replaced_count = 0
    
    def _replace(module):
        nonlocal replaced_count
        
        # 1. Sequential / ModuleList 컨테이너 처리 (인덱스를 통한 접근)
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for i in range(len(module)):
                child = module[i]
                
                # 가로채기(Monkey-patch)로 인해 nn.Conv2d가 변조되었을 수 있으므로 Native 클래스로 확인합니다.
                if isinstance(child, NativeConv2d) and not isinstance(child, quant_nn.QuantConv2d):
                    new_module = quant_nn.QuantConv2d(
                        child.in_channels, child.out_channels, child.kernel_size,
                        child.stride, child.padding, child.dilation, child.groups,
                        child.bias is not None, child.padding_mode
                    )
                    new_module.weight.data.copy_(child.weight.data)
                    if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                    module[i] = new_module
                    replaced_count += 1
                elif isinstance(child, NativeLinear) and not isinstance(child, quant_nn.QuantLinear):
                    new_module = quant_nn.QuantLinear(child.in_features, child.out_features, child.bias is not None)
                    new_module.weight.data.copy_(child.weight.data)
                    if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                    module[i] = new_module
                    replaced_count += 1
                else:
                    _replace(child)
            return

        # 2. 일반 Module 객체들 처리 (named_children 순회 방식)
        for name, child in module.named_children():
            if isinstance(child, NativeConv2d) and not isinstance(child, quant_nn.QuantConv2d):
                new_module = quant_nn.QuantConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, child.groups,
                    child.bias is not None, child.padding_mode
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                setattr(module, name, new_module)
                replaced_count += 1
            elif isinstance(child, NativeLinear) and not isinstance(child, quant_nn.QuantLinear):
                new_module = quant_nn.QuantLinear(child.in_features, child.out_features, child.bias is not None)
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                setattr(module, name, new_module)
                replaced_count += 1
            else:
                _replace(child)

    _replace(model)
    print(f"✅ 레이어 교체가 성공적으로 수행되었습니다. (교체된 총 개수: {replaced_count}개)")

def disable_sensitive_layers_quantization(model: nn.Module) -> None:
    """Detect Head 계층이나 Attention 블록처럼 표현력이 중요한 민감 계층의 양자화를 우회시킵니다."""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    # TensorRT 변환 장애나 mAP의 치명적인 손실을 야기할 수 있는 패턴 목록입니다.
    sensitive_patterns = ['detect', 'head', 'attn']
    
    print(f"🛡️  안정성 확보를 위해 일부 민감 레이어들의 양자화를 강제 비활성화합니다 (대상: Detect, Head, Attn)...")
    disabled_count = 0
    
    # 헬퍼 함수: 특정 단일 모듈을 비활성화
    def _disable_module(m):
        did_disable = False
        if isinstance(m, (quant_nn.QuantConv2d, quant_nn.QuantLinear)):
            if hasattr(m, '_input_quantizer'):
                m._input_quantizer.disable()
                m._input_quantizer._explicitly_disabled = True
            if hasattr(m, '_weight_quantizer'):
                m._weight_quantizer.disable()
                m._weight_quantizer._explicitly_disabled = True
            did_disable = True
        return did_disable

    for name, module in model.named_modules():
        # 1. 이름 기반(Name) 패턴 매칭
        if any(p in name.lower() for p in sensitive_patterns):
             if _disable_module(module): disabled_count += 1
             
        # 2. 타입(Class Type) 기반 매칭
        # 설계 구조 상 순차적(Sequential) 모델 레이어들의 이름이 단순 숫자("23")인 문제를 해결합니다.
        # Ultralytics의 Detect/Segment/Pose/Classify 특수 헤더 구별에 치명적으로 작용합니다.
        class_name = module.__class__.__name__.lower()
        if class_name in ['detect', 'segment', 'pose', 'classify', 'head']:
            print(f"   알림: 이름이 아닌 '타입'을 기반으로 민감 계층을 탐지했습니다: {class_name} ({name})")
            # 탐지된 헤더 계층 아래의 모든 자식 계층을 재귀적으로 비활성화합니다.
            for child in module.modules():
                if _disable_module(child): disabled_count += 1

    print(f"✅ 총 {disabled_count}개의 민감 계층 모듈들을 양자화 대상에서 제외시켰습니다.")

def collect_calibration_stats(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_batches: int = 64,
    calib_method: str = 'mse',
    device: str = 'cuda'
) -> None:
    """활성화 값(Activation)의 범주를 수집하기 위하여 모델 보정(Calibration)을 실행합니다."""
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import calib
    except ImportError:
        return

    print(f"📊 보정(Calibration) 프로세스를 개시합니다... (기법: {calib_method}, 허용 배치 수: {num_batches})")
    model.eval()
    model.to(device)

    # 1. 전역 보정 모드 켜기 (Enable Calibration)
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_explicitly_disabled', False):
                continue
            
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()

    # 2. 데이터 유입 및 측정 시작
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, total=num_batches, desc="최적 활성화 구간 보정 중")):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                images = batch.get('img', batch.get('image'))
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device).float() / 255.0
            model(images)

    # 3. 보정 모드를 해제하고 측정된 Amax(최대절대값) 확정하기
    print(f"📉 측정된 로그를 바탕으로 {calib_method} 방식에 따라 권장 Amax를 도출하고 계산합니다...")
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_explicitly_disabled', False):
                continue

            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
                
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module.load_calib_amax(method=calib_method, strict=False)
                else:
                    module.load_calib_amax(strict=False)
                
                module._calibrator = None

    print("✅ 모델 보정이 완료되었습니다.")

def prepare_model_for_export(model: nn.Module) -> None:
    """양자화 모듈들을 활성화로 고정하고, 전체 Quantizer들이 ONNX로 Export 되도록 'use_fb_fake_quant' 플래그를 할당합니다."""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    print("🔧 ONNX 추출 모드를 고정합니다 (양자화 및 use_fb_fake_quant 활성화)...")
    
    # [핵심] 훅(Hooks)이 올바르게 등록될 수 있도록 클래스 전역 스코프에 설정해야 합니다.
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # 사용자가 비활성화 설정한 계층(예: 민감 계층 Head 부분)은 순수 FP32를 유지해야 하므로 건너뜁니다.
            if getattr(module, '_explicitly_disabled', False):
                continue

            # [핵심] 여기서 강제로 enable_quant() 를 실행하면 안 됩니다!
            # 이것은 'disable_sensitive_layers_quantization'의 조치를 무효화시키며, 
            # 보정조차 안된 헤더 계층들이 엉터리로 양자화되는 결과를 초래합니다.
            # module.enable_quant() 
            
            # 내보내기용 특수 플래그만 부여합니다.
            module.use_fb_fake_quant = True
            
            # [버그 픽스] ONNX Export를 위해 _amax 값이 스칼라이거나 일관된 형태인지 검증합니다.
            # 일부 Opset 13 환경에서는 amax가 특정하게 명시된 형태 외의 0차원 텐서일 경우 중단(Crash)이 발생합니다.
            if module._amax.numel() == 1:
                 module._amax.data = module._amax.data.reshape([])
            
            # 확증된 amax가 있는 경우에 한해서 내보내기 플래그 덧씌움
            module.use_fb_fake_quant = True


def get_calibration_dataloader(
    data_yaml_path: str,
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 4
):
    """
    모든 데이터 증강(Augmentations) 기법들이 무효화(DISABLED)된 매우 정순한 모델 보정 전용 데이터 로더를 구축합니다.
    이 조치는 Activation 구간 분석 및 Amax 수치 연산이 왜곡되지 않도록 하는데 있어 매우 중요합니다.
    """
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.dataset import YOLODataset
    import yaml
    from ultralytics.utils import DEFAULT_CFG
    from types import SimpleNamespace
    
    # 1. 데이터(yaml) 설정 가져오기
    data_cfg = check_det_dataset(data_yaml_path)
    train_path = data_cfg['train']
    if isinstance(train_path, list): train_path = train_path[0]
    
    # 2. 증강 없이 바로 Dataset 모델 생성 
    # 기본 강제된 증강 기법을 회피하고자 모델 Trainer 캡슐화를 우회하여 직접 생성(YOLODataset)합니다.
    # [수정안] DEFAULT_CFG는 이미 IterableSimpleNamespace의 파생이므로 바로 전달해도 됩니다.
    hyp_obj = DEFAULT_CFG
    
    dataset = YOLODataset(
        img_path=train_path,
        imgsz=imgsz,
        batch_size=batch_size,
        augment=False,       # <--- 핵심 조치: Mosaic 및 Mixup 데이터 변형 증강 기법 원천봉쇄
        hyp=hyp_obj,         # <--- 조치사항: 객체 래퍼 그대로 전달
        rect=False,          # 추론 때의 직사각형이 아닌 표준 형태 유도
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        data=data_cfg,
        classes=data_cfg['names']
    )
    
    # 3. 로더 구축 (Builder)
    from torch.utils.data import DataLoader
    # Ultralytics의 원시 collate_fn을 활용합니다.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,        # 데이터 구조 자체는 깨끗하므로 섞는 것은 무방합니다.
        num_workers=workers,
        collate_fn=getattr(dataset, 'collate_fn', None),
        pin_memory=True
    )
    
    return loader
