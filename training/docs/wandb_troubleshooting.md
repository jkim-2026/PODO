# WandB 연동 트러블슈팅 기록

> 작성일: 2026-02-23  
> 환경: RunPod (NVIDIA RTX 4090, Ubuntu, Python 3.11)  
> 목표: `scripts/run_exp.py` 실행 → WandB에 학습 로그 및 모델 아티팩트 업로드 확인

---

## 최종 실행 명령어

```bash
# training/ 디렉토리 안에서 실행해야 함
cd /workspace/pro-cv-finalproject-cv-01/training
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml
```

---

## 발생한 에러 목록 및 해결 과정

---

### 에러 1: `No such file or directory`

**에러 메시지**
```
/venv/bin/python3: can't open file '/workspace/pro-cv-finalproject-cv-01/scripts/run_exp.py': 
[Errno 2] No such file or directory
```

**원인**  
CWD(현재 작업 디렉토리)가 프로젝트 루트(`/workspace/pro-cv-finalproject-cv-01`)인 상태에서 `scripts/run_exp.py`를 상대 경로로 참조했기 때문.  
실제 스크립트는 `training/scripts/run_exp.py`에 위치한다.

**해결**  
- 항상 `training/` 폴더 안으로 이동한 뒤 실행하거나
- 전체 경로를 명시해서 실행

```bash
# 방법 A: 폴더 이동 후 실행 (권장)
cd training
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml

# 방법 B: 전체 경로 지정
/root/.local/bin/uv run --project training python training/scripts/run_exp.py --config training/configs/config.yaml
```

**고려할 점**  
- `uv run`은 `--project` 플래그로 프로젝트 루트를 지정해도, Python 인터프리터의 CWD는 쉘의 CWD를 따른다.
- `scripts/run_exp.py` 내부에서 `os.path.dirname(__file__)` 기반으로 경로를 계산하므로, 스크립트 위치 기준의 상대 경로 탐색은 정상 동작한다.

---

### 에러 2: `Disk quota exceeded`

**에러 메시지**
```
OSError: [Errno 122] Disk quota exceeded
```

**원인**  
볼륨(20GB)이 꽉 찬 상태. 원인은 두 개의 가상환경이 중복 생성되었기 때문.

| 경로 | 크기 |
|------|------|
| `/workspace/pro-cv-finalproject-cv-01/.venv` | **8.6 GB** |
| `/workspace/pro-cv-finalproject-cv-01/training/.venv` | **10 GB** |

`uv pip install wandb`를 루트 경로에서 실행하여 바깥쪽에 불필요한 가상환경이 생성되었다.

**해결**  
바깥쪽 중복 가상환경 삭제
```bash
rm -rf /workspace/pro-cv-finalproject-cv-01/.venv
```

**고려할 점**  
- `uv` 명령 실행 시 항상 어느 경로에서 실행하는지 확인할 것.
- `training/` 폴더에만 `.venv`가 있어야 한다.
- 볼륨 부족 시 `uv cache clean`으로 캐시를 정리할 수 있다 (설치 완료 후엔 삭제해도 무방).
  ```bash
  /root/.local/bin/uv cache clean
  # → 최대 14GB+ 절약 가능
  ```
- `df -h`의 용량과 실제 RunPod 대시보드의 Volume 용량이 다를 수 있으므로, 동시에 두 가지를 확인할 것.

---

### 에러 3: `PytorchStreamReader failed reading zip archive`

**에러 메시지**
```
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

**원인**  
디스크가 꽉 찬 상태에서 `yolo11x.pt` 가중치 파일을 다운로드하다 중단되어 파일이 손상(corrupt)됨.  
정상 파일 크기: **~114 MB** → 손상된 파일 크기: **64 MB**

```bash
# 손상 확인
ls -lh training/yolo11x.pt
# 64MB이면 비정상
```

**해결**  
1. 손상된 파일 삭제
2. `pretrained_weights/` 폴더에 재다운로드

```bash
# 손상된 파일 삭제
rm -f yolo11x.pt

# pretrained_weights 폴더 생성 후 재다운로드
mkdir -p pretrained_weights
./.venv/bin/python -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolo11x.pt')  # 자동 다운로드
shutil.move('yolo11x.pt', 'pretrained_weights/yolo11x.pt')
print('Done')
"
```

**고려할 점**  
- 디스크 용량 부족 상황에서 다운로드가 중단되어도 오류 메시지 없이 파일만 불완전하게 저장될 수 있다.
- 모델 로드 시 항상 파일 크기를 먼저 확인하는 습관이 필요하다.
- `src/models/yolov11x.py`에서 가중치 탐색 순서:
  1. 현재 경로(`./yolo11x.pt`)
  2. `pretrained_weights/yolo11x.pt`
  3. Ultralytics 자동 다운로드

---

### 에러 4: `Invalid project name: cannot contain '/'`

**에러 메시지**
```
wandb.errors.UsageError: Invalid project name 'ckgqf1313-boostcamp/PODO': 
cannot contain characters '/,\,#,?,%,:', found '/'
```

**원인**  
`config.yaml`의 `wandb_project`에 `entity/project` 형식(`"ckgqf1313-boostcamp/PODO"`)으로 저장되어 있으나, 기존 코드에서 이를 `project=` 인자 하나로 통째로 넘겼기 때문.

**기존 코드 (버그)**
```python
# src/train.py
wandb.init(project=self.config['wandb_project'], ...)
# → project='ckgqf1313-boostcamp/PODO' (슬래시 포함 → 에러)
```

**수정된 코드**
```python
# src/train.py
wandb_project_str = self.config['wandb_project']
if '/' in wandb_project_str:
    wandb_entity, wandb_project = wandb_project_str.split('/', 1)
else:
    wandb_entity, wandb_project = None, wandb_project_str

wandb.init(
    entity=wandb_entity,    # 'ckgqf1313-boostcamp'
    project=wandb_project,  # 'PODO'
    name=f"{self.config['exp_name']}",
    reinit=True
)
```

**고려할 점**  
- `config.yaml`의 `wandb_project`는 `entity/project` 형식으로 유지하되, 코드에서 파싱해야 한다.
- `entity`는 WandB의 팀/조직 이름, `project`는 프로젝트 이름이다.
- `entity`를 `None`으로 넘기면 기본 개인 계정으로 로깅된다.

---

### 에러 5: `401 Unauthorized - user is not logged in`

**에러 메시지**
```
wandb.errors.CommError: Error uploading run: returned error 401: 
{"errors":[{"message":"user is not logged in"}]}
```

**원인**  
`wandb login`을 정상적으로 실행했으나, `/root/.netrc` 파일에 API 키가 **두 번 중복**으로 저장된 채 손상됨.

```
# 손상된 .netrc
password wandb_v1_QD2phLZ...wandb_v1_QD2phLZ...  ← 키가 2번 붙어있음
```

**해결**  
`~/.netrc` 파일을 직접 열어 올바른 형식으로 수정

```
machine api.wandb.ai
  login user
  password wandb_v1_XXXXXXXXXXXXXXXXXXXXXXXX  ← 키를 한 번만 작성
```

또는 `wandb login`을 다시 실행할 때 중단하지 말고 완료시킬 것.
```bash
wandb login --relogin
```

**고려할 점**  
- `wandb login` 실행 중 `Ctrl+C`로 중단하면 키가 반만 저장될 수 있다.
- API 키 확인 방법: https://wandb.ai/authorize
- 환경변수로도 인증 가능 (`.netrc` 없이도 작동):
  ```bash
  export WANDB_API_KEY=wandb_v1_XXXX...
  ```

---

## 수정된 파일 목록

| 파일 | 수정 내용 |
|------|-----------|
| `src/train.py` | `wandb.init()` 호출 시 `entity/project` 형식을 파싱하여 `entity`와 `project`를 분리해서 전달하도록 수정 |
| `/root/.netrc` | 중복 저장된 WandB API 키를 정상 형식으로 복원 |

---

## 향후 실행 시 체크리스트

학습 실행 전 아래 항목을 순서대로 확인할 것.

```bash
# 1. 의존성 설치 확인 (최초 1회 또는 환경 변경 시)
cd /workspace/pro-cv-finalproject-cv-01/training
/root/.local/bin/uv sync

# 2. 디스크 용량 확인
df -h .
du -sh .venv PCB_DATASET runs

# 3. 가중치 파일 정상 여부 확인
ls -lh pretrained_weights/yolo11x.pt
# → 110MB 이상이어야 정상

# 4. WandB 로그인 확인
./.venv/bin/wandb status
# OR
./.venv/bin/python -c "import wandb; wandb.login(); print('OK')"

# 5. 학습 실행
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml
```

---

## WandB 프로젝트 구조

```
WandB Project: ckgqf1313-boostcamp/PODO
├── Runs      → 각 실험의 학습 로그 (loss, mAP 등)
└── Artifacts → best.pt 모델 파일 (학습 완료 후 자동 업로드)
               tags: ['latest', 'production']
```

학습 완료 후 WandB 대시보드에서 확인:  
👉 https://wandb.ai/ckgqf1313-boostcamp/PODO
