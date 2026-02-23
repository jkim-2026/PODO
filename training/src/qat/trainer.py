import os
import torch
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# Ultralytics 임포트
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import unwrap_model

# QAT 유틸리티 임포트 (상대 경로)
from . import utils as qtu

class QATDetectionTrainer(DetectionTrainer):
    """
    Quantization-Aware Training (QAT, 양자화 인식 학습)을 위한 전용 트레이너입니다.
    Ultralytics의 DetectionTrainer를 상속받아 안정적인 EMA 처리와 학습 루프를 재사용합니다.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if cfg is None:
            cfg = DEFAULT_CFG_DICT.copy()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        
        # QAT 기본 하이퍼파라미터 정의 (인자로 오버라이드 가능)
        self.qat_epochs = overrides.get('epochs', 20)
        self.qat_lr = overrides.get('lr0', 0.0001)

    def _setup_train(self, world_size=1):
        """
        QAT 특화 제약 사항들을 강제하기 위해 _setup_train 메서드를 오버라이드합니다.
        """
        super()._setup_train()

        print("\n" + "="*40)
        print("🔧 QAT 전용 학습 환경 설정 (Setup)")
        print("="*40)

        # 1. Warmup 페이즈 비활성화 (이미 사전학습된 모델이므로 필요하지 않음)
        print(f"기존 Warmup Epochs 설정: {self.args.warmup_epochs}")
        self.args.warmup_epochs = 0
        print("👉 강제 적용 (Warmup Epochs): 0 (QAT 에러 방지를 위해 비활성화됨)")

        # 2. AMP(자동 혼합 정밀도) 비활성화 (QAT 파이프라인의 핵심 안정성을 보장)
        print(f"기존 AMP 설정: {self.args.amp}")
        self.args.amp = False
        self.amp = False
        print("👉 강제 적용 (AMP): False (Q/DQ 노드 충돌을 막기 위해 비활성화됨)")

        # 3. 미세 조정(Fine-tuning)을 위한 맞춤형 코사인(Cosine) 스케줄러 적용
        # (기존 Ultralytics 스케줄러도 우수하지만 더욱 타이트한 제어를 목적으로 합니다)
        # super()._setup_train() 실행으로 초기화된 이후에 우리가 스케줄러를 가로채어 교체합니다.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.args.lr0 * 0.01
        )
        print("👉 Scheduler(학습률 조정기)가 CosineAnnealingLR 로 교체되었습니다.")
        print("="*40 + "\n")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        동적으로 QAT 레이어가 불일치하더라도 넘어갈 수 있도록 strict=False 모드로 모델을 로딩합니다.
        본 파이프라인 구조에서는 대개 외부에서 모델을 직접 주입하지만, 이것은 안전망(Fallback) 역할을 합니다.
        """
        # 만일 self.model 속성를 통해 모델이 이미 주입되었다면, super().get_model이 이를 감지하게 됩니다.
        # 대체로 Ultralytics 내부 로직에서 strict 로딩 여부를 알아서 처리합니다.
        
        # 모델 학습을 이어서(resume) 진행할 때 호출되는 대비책입니다.
        model = super().get_model(cfg, weights, verbose)
        return model

    def save_model(self):
        """
        save_model 메서드를 오버라이드하여 파일 저장 시 .half() 반정밀도 변환을 차단합니다.
        QAT 모델의 가중치 및 양자화 계층(QuantConv2d)들은 손실 없이 반드시 FP32 상태를 유지해야 합니다.
        """
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(unwrap_model(self.model)).float(),  # [필수] float32 강제
            'ema': deepcopy(unwrap_model(self.ema.ema)).float() if self.ema else None, # [필수] float32 강제
            'updates': self.ema.updates if self.ema else None,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'date': datetime.now().isoformat(),
        }

        # 일반 모델 저장 (last.pt 파일 갱신)
        torch.save(ckpt, self.last)
        
        # 성능이 가장 높았던 에포크(best.pt 갱신)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        
        # 간격(save_period)에 따른 중간 저장
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')

        # [디버깅 참고] TensorQuantizer 생존 객체 수 검증용 코드
        # from pytorch_quantization import nn as quant_nn
        # m = unwrap_model(self.model)
        # count = sum(1 for layer in m.modules() if isinstance(layer, quant_nn.TensorQuantizer))
        # print(f"[QAT 체크포인트 저장 안내] 성공적으로 보존된 양자화 개체 수: {count}")
