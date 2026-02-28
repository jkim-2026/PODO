# 모델 학습

PCB 결함 탐지 YOLO 모델 학습 모듈입니다.

## 개요

**역할:** YOLOv11m 모델을 PCB 결함 데이터셋으로 학습

## 기술 스택

| 구분 | 기술 |
|------|------|
| Python | 3.8 이상 |
| 모델 | YOLOv11m |
| 학습 기법 | QAT (Quantization-Aware Training), KD (Knowledge Distillation) |
| 자동화 | Apache Airflow (MLOps Pipeline) |
| 패키지 관리 | uv |

## 프로젝트 구조

```
training/
├── configs/                 # 모델 학습 및 KD 설정 (config.yaml 등)
├── dags/                    # Airflow MLOps 재학습/배포 DAG
├── scripts/
│   ├── run_exp.py           # 베이스라인 메인 실행
│   ├── train_qat.py         # QAT 학습 실행
│   ├── train_kd.py          # 지식 증류(KD) 학습 실행
│   └── export_qat.py        # TensorRT (Engine/ONNX) 자체 Export
├── src/
│   ├── datasets/            # 데이터셋 로드 및 분석
│   ├── models/              # YOLO 모델 래퍼
│   ├── qat/                 # QAT 관련 모듈 (EMA, Hook 등)
│   ├── utils/               # 유틸리티
│   ├── train.py             # 트레이너 클래스
│   └── inference.py         # 추론 매니저
└── pretrained_weights/      # 사전 학습 가중치 (Teacher 모델 등)
```

## 데이터셋

**PCB_DATASET:**
- 클래스: 6종 (scratch, hole, contamination, crack 등)
- 포맷: XML 라벨 → YOLO 포맷 자동 변환
- 분할: Stratified Split (7:2:1)

## 모델

**YOLOv11m:**
- 사전 학습 가중치: `yolov11m.pt`
- Edge 배포 최적화: QAT 모델 배포 (`yolov11m_v1.engine` 등)

## 학습 설정 (`configs/config.yaml`)

```yaml
# 실험 이름
exp_name: "baseline_v11m"

# 모델 설정
model_module: "yolo"
model: "yolo11m"

# 학습 파라미터
epochs: 180
optimizer: 'AdamW'
scheduler_type: 'cosine'
lrf: 0.1
device: '0'
seed: 42

# 가중치 (Loss Gains)
cls: 2.0    # 분류 성능 우선 (mAP50)

# 증강 제어
mosaic: 1.0
scale: 0.5
mixup: 0.0
```

## 평가 지표

## 평가 지표

배포 우선순위는 다음과 같습니다.

| 지표 | 설명 | 비고 |
|------|------|------|
| **Recall** | 실제 결함 중 검출 성공 비율 | 불량품 유출 방지를 위해 **Recall 1.0 최우선 검증** |
| FPS | 실시간 추론 속도 | Jetson Orin Nano 기준 **30 FPS 이상** 타겟 |
| mAP50 | IoU 0.5에서의 평균 정밀도 | 모델 간 절대 성능 비교 지표 |
| mAP50-95 | IoU 0.5~0.95 평균 정밀도 | 위치 정확도를 포함한 보조 지표 |

## 실행 방법

### 의존성 설치

```bash
cd training
uv sync
```

### 가중치 준비

사전 학습 모델 가중치를 `pretrained_weights/` 폴더에 넣습니다. (예: `yolo11m.pt`)

### 학습 실행

**1. 일반 베이스라인 학습 (Fine-Tuning)**

```bash
# 기본 설정으로 실행
python scripts/run_exp.py --config configs/config.yaml

# uv를 통한 실행
uv run python scripts/run_exp.py --config configs/config.yaml
```

**2. QAT (Quantization-Aware Training) 모델 및 엔진 빌드**

Edge 장비의 실시간 배포를 위한 INT8 모델 양자화 및 TensorRT(Engine) 변환 과정입니다.

```bash
# 1단계. QAT 학습 (커스텀 EMA 트레이너 적용)
python scripts/train_qat.py --config configs/config.yaml

# 2단계. 양자화 통계 재보정 (Re-calibration)
python scripts/recalibrate_ema.py --weights runs/baseline_v11m_qat/weights/best_ema.pt --data PCB_DATASET/data.yaml

# 3단계. 커스텀 레이어 호환 ONNX 모델 및 엔진 추출 (Deep Injection 우회 변환)
python scripts/export_qat.py --weights runs/baseline_v11m_qat/weights/best_ema_recalib.pt
```

**단일 자동 진행 과정:**
1. 데이터 준비 (XML → YOLO 변환, 7:2:1 분할)
2. 모델 성능 학습 (베이스라인 또는 QAT, KD)
3. 모델 압축 및 추출 (QAT/Engine Export)
4. 검증 및 추론 (테스트 세트 평가)

## 결과물

모든 실험 결과물은 `runs/{exp_name}/` 폴더와 `mlruns/` 폴더에 분산되어 저장됩니다:

| 항목 | 경로 | 설명 |
|------|------|------|
| 모델 가중치 | `runs/{exp_name}/weights/best.pt` | 모델을 재학습한 결과 최고의 성능 가중치 |
| 추론 결과 | `runs/{exp_name}/inference/submission.csv` | 테스트 폴더에 대한 최종 분류값 파일 |
| 시각화 | `runs/{exp_name}/detect/` | 이미지 결과 |
| 로그 및 메트릭 | 로컬 MLflow (`mlruns/`) | 학습 과정의 모든 주요 메트릭 및 모델 아티팩트 레지스트리 |

## 확장성

### 새로운 데이터셋 전략

`src/datasets/my_dataset.py`를 만들고 `get_dataset(config)` 함수를 구현한 후, `config.yaml`에서 `dataset_module`을 변경합니다.

### 새로운 모델

`src/models/my_model.py`를 만들고 `get_model(config)` 함수를 구현한 후, `config.yaml`에서 `model_module`을 변경합니다.

## 상세 문서

전체 상세 구현은 `training/README.md` 참조.
