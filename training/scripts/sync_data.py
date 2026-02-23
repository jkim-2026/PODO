import os
import sys
import yaml
import shutil
import random
import boto3
from pathlib import Path
from tqdm import tqdm

from tqdm import tqdm

# 프로젝트 루트 디렉토리를 탐색 경로에 추가하여 자체 모듈간 상호작용 및 절대 참조를 수월하게 구성
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# AWS나 시스템의 환경변수 구조 세팅을 읽습니다 (API 서버 구성품의 config 양식과 흡사하게 연동한다고 가정합니다)
# 단순성을 보장하기 위하여 yaml 또는 env 탐색을 시도하지만 여의치 않다면 하드코딩 대체 또는 파라미터를 통할 수 있게 끔 구성합니다.
# 가장 바람직한 구조는 모델 훈련 API와 이 구성 로직이 단극화된 객체를 함께 가져다 공유하며 쓰는 방식입니다.
try:
    from config import settings
except ImportError:
    # API 환경의 동등한 변수 통제가 부재할 때 스스로 임시 더미 환경변수 뼈대를 자체 조형(Fallback)하여 사용합니다.
    class Settings:
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
        S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "pcb-data-storage")
        DATASET_DIR = os.path.join(project_root, "PCB_DATASET")
    settings = Settings()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

def sync_refined_data(local_dataset_msg_dir: str):
    """
    S3 원격 클라우드 저장소 공간의 refined/images 와 refined/labels 버킷 항목에서 새로운 결괏값을 병합 스캔하고 다운로드 받습니다.
    새로 발견되어 물리적으로 동기화 갱신된 파일들의 확장자 제외 중심 이름명(Stem) 리스트 항목을 리턴하여 반환합니다.
    """
    bucket = settings.S3_BUCKET_NAME
    prefix = "refined/"
    
    # 다운로드를 실행하기 이전 로컬 환경을 분석하고 기본 저장 구실 역할을 지닌 디렉터리를 반드시 실존하도록 확립해둡니다.
    images_dir = Path(local_dataset_msg_dir) / "images"
    labels_dir = Path(local_dataset_msg_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 s3://{bucket}/{prefix} 외부 버킷의 객체와 병합 교환 다운로드를 실시하며 탐색 중입니다...")
    
    # refined/ 하위의 모든 객체를 페이지네이션 검색 기반으로 순진 조회합니다.
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    new_files = []
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            
            # 항목이 비어있는 일반 디렉토리 구조 껍데기 폴더 자체라면 처리하지 않고 곧장 스킵합니다.
            if not filename: 
                continue

            # 그것이 올바른 이미지 확장자를 가진 대상이거나 라벨 확장자 타겟이 맞는지 재분석 규명 작업을 실시합니다.
            if key.startswith("refined/images/") and (filename.endswith('.jpg') or filename.endswith('.png')):
                target_path = images_dir / filename
                if not target_path.exists(): # 이전에 로컬 환경에 다운로드 또는 구비 된 적이 없을 때만 희귀 자원으로 취급받아 통신을 전송해 다운로드
                    s3_client.download_file(bucket, key, str(target_path))
                    new_files.append(target_path.stem)
                    # 위 이미지 대응쌍과 호환이 완벽히 매칭된 동일 짝지어진 레이블 문서(텍스트 파일)의 쌍끌이 파싱 동반 다운로드 시도
                    label_key = key.replace("images/", "labels/").replace(Path(key).suffix, ".txt")
                    label_path = labels_dir / (Path(filename).stem + ".txt")
                    try:
                        s3_client.download_file(bucket, label_key, str(label_path))
                    except Exception:
                        print(f"⚠️ 경고 통지: {filename} 와 직접적으로 짝 지어진 주석 라벨 txt 파일을 로드하지 못했거나 찾기 실패.")
            
    print(f"✅ 원격 보정 파싱 결과 총괄 {len(new_files)} 개의 진척 된 새 이미지들(New Image)을 원활하게 성공적으로 다운로드 받았습니다.")
    return new_files

def load_existing_splits(dataset_dir):
    """
    주어진 디렉터리 시스템 하위에 train.txt나 val.txt 등의 기록문서를 검토해 이미 로컬 내부 모델로 레지스터화(등록완료)되어 묶여 활용 처리된 적 있던 모든 이미지들의 배열들을 식별합니다.
    Returns: 완벽하게 정리되어 신뢰할 수 있게 비교 호환 검증된 절대 경로들의 고유 셋 모음 (Set of absolute paths or stems)
    """
    existing_images = set()
    for split in ["train.txt", "val.txt", "test_images.txt"]:
        path = Path(dataset_dir) / split
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    # 명백하고 의심의 여지가 없이 아주 견고무결하게 비교를 할 수 있게끔 언제나 완벽한 절대경로 스키마 형태 상태 자체를 추출 저장 구조로 변모해 반영시킵니다.
                    existing_images.add(line.strip())
    return existing_images

def create_transient_datasets(dataset_dir, new_stems):
    """
    지속적인 자동 재학습 훈련(Retraining Pipeline)을 타겟으로 하여 다음의 임시성 데이터셋 설정 파일과 분할된 메타 스플릿 구성품 양식을 일시 창설 조형 제조합니다:
    - train_retrain.txt = 기존 학습 train.txt 파일 통계 + 새롭게 검출된 추가 데이터 분량 (new_train, 약 90%)
    - val_retrain.txt = 과거 평가 val.txt 파편 정보 + 금번에 새롭게 편입된 신예 데이터 파편 중 할당량 (new_val, 약 10%)
    - data_retrain.yaml = 이들 새로 발급된 신규 교본 통계 데이터 파일들을 가이드 참조 가리키는 역할의 관제 포인터
    """
    # 1. 초기 원본 오리지널 분할 데이터 세트 배열을 탑재하여 적재합니다.
    train_orig = []
    val_orig = []
    
    if (Path(dataset_dir) / "train.txt").exists():
        with open(Path(dataset_dir) / "train.txt", "r") as f:
            train_orig = [line.strip() for line in f if line.strip()]
            
    if (Path(dataset_dir) / "val.txt").exists():
        with open(Path(dataset_dir) / "val.txt", "r") as f:
            val_orig = [line.strip() for line in f if line.strip()]

    print(f"📉 기본 베이스(Base) 오리지널 참조 기저 데이터셋 현황: {len(train_orig)} Train 세트, {len(val_orig)} Val 검증 세트")

    # 2. 추가적으로 발굴된 신규 유입군 데이터 대상물 세트 물량 자원을 (9:1) 비율 체제로 균등 스플릿 전개 분할시킵니다.
    if new_stems:
        random.shuffle(new_stems)
        split_idx = int(len(new_stems) * 0.9)
        new_train_stems = new_stems[:split_idx]
        new_val_stems = new_stems[split_idx:]
        
        print(f"🆕 새로 추가된 데이터 통계 현황 총론: {len(new_train_stems)} Train 학습 자원 세트 집행, {len(new_val_stems)} Val 평가 검증 지표 자원 합류")
    else:
        new_train_stems = []
        new_val_stems = []
        print("ℹ️  어떤 무결한 신규 자원으로 추가 규합 판정 지어진 요소 군락 객체 배열군이 이전에 아예 없었습니다. 따라서 기존 낡은 베이스 통계 구성만 재사용 합니다.")

    # 스템(확장자 미포함 기본 깡통 문서 원형 이름) 정보 값만을 이용해서 하드웨어적인 실제 경로 전체 풀 문자열 지표를 다시 계산 추출 반환 해주는 은닉 지원성 유틸리티 함수 스왑 모듈
    def stems_to_paths(stems):
        paths = []
        for s in stems:
            # 타겟 다운로드 베이스 경로(Root) 구조 하위 어느곳이든간에 혹시 짱박혀있을 하위 디렉터리나 파일 폴더의 실낱같은 희망 잔존 확률까지 모조리 다 스쳐 색적 탐사하기 위한 glob 전위 탐색 병행 기동 수사 수행
            found = list(Path(dataset_dir).rglob(f"{s}.jpg"))
            if found:
                paths.append(str(found[0].absolute()))
        return paths

    new_train_paths = stems_to_paths(new_train_stems)
    new_val_paths = stems_to_paths(new_val_stems)

    # 3. 신/구 통계 파일 정보 객체 주소 연계 병합 수립 결합(Combine)
    train_final = train_orig + new_train_paths
    val_final = val_orig + new_val_paths

    # 4. 임시 소모성 샌드박스 보존 파일 문서들(Transient Files)들을 작성 기입 하여 반영 처리합니다.
    retrain_train_path = Path(dataset_dir) / "train_retrain.txt"
    retrain_val_path = Path(dataset_dir) / "val_retrain.txt"

    with open(retrain_train_path, "w") as f:
        f.write("\n".join(train_final))
    
    with open(retrain_val_path, "w") as f:
        f.write("\n".join(val_final))

    print(f"✅ 임시 변칙 재훈련용 양분 스플릿 구조 설계 재편 재구축 전개 병합 조립 완료: {len(train_final)} Train 데이터, {len(val_final)} Val 데이터")
    print(f"   -> {retrain_train_path}")
    print(f"   -> {retrain_val_path}")

    # 5. 이러한 일련의 임시 구도 과정을 하나로 통제 관할 해줄 별개의 트랜지언트(Transient)용 특수 가상 데이터 환경변수 yaml 세팅 파일 창시 공정
    # 우선 오리지널 양식의 data.yaml 문서를 해부해서 적힌 클래스 메타 이름(이름 및 속성 계층 목록)들 배열 구조체 표본 수치를 사전 스틸 복각해야만 합니다.
    orig_yaml_path = Path(dataset_dir) / "data.yaml"
    names = {}
    nc = 0
    
    if orig_yaml_path.exists():
        with open(orig_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            names = data_config.get('names', {})
            nc = data_config.get('nc', len(names))
    else:
        # 대비책(Fallback): 기본 설정에 정의된 PCB의 하드코딩된 원석 클래스 식별 결함 종류 표본들을 강권 발동하여 기본 규격(Default)으로써 조립시킵니다 (config.settings.CLASS_ID_MAP 참조와 유사 동작)
        names = {
            0: "missing_hole",
            1: "mouse_bite",
            2: "open_circuit",
            3: "short",
            4: "spur",
            5: "spurious_copper"
        }
        nc = 6
        print("⚠️  data.yaml 파일이 탐지되지 않았으므로 어쩔수없이 기본 하드코딩된 강제 PCB 불량 감지 모델 결함 레이블 군 목록 클래스 이름 정보 시스템을 오버라이딩 채택 주입 했습니다.")
    
    retrain_yaml_content = {
        'path': str(dataset_dir),
        'train': str(retrain_train_path.absolute()),
        'val': str(retrain_val_path.absolute()),
        'nc': nc,
        'names': names
    }

    retrain_yaml_path = Path(dataset_dir) / "data_retrain.yaml"
    with open(retrain_yaml_path, 'w') as f:
        yaml.dump(retrain_yaml_content, f)
        
    print(f"📄 새로운 훈련 데이터 참조 포인터 가상 설정 yaml 파일이 탄생 되었습니다: {retrain_yaml_path}")
    return str(retrain_yaml_path)

def main():
    dataset_dir = settings.DATASET_DIR
    
    # 1. 파일 동기화 시스템 (Sync) (클라우드 파일을 다운로드 하고 베이스 원본 데이터 집합에는 미처 소속 등록이 안되어져 있는 모든 외부의 새로운 미아 Stem 리스트를 디스크 위에서 추출 도출 귀결 산정하는 매커니즘 과정 처리)
    # 로직 재정의 이행 결과: 이제 'sync_refined_data' 함수는 단지 순수무결하게 새로운 통신 파싱 파일들을 맹목 다운로드 로컬로 퍼오는 작업만을 합니다.
    # 하지만 우리 시스템은 이 파싱 파일들이 현재의 기존 기 보유 데이터 수치량 기준에서 진실로 'New' 데이터 집합체로 규정 합치되는지에 대한 재귀적인 분별 증명 연산 과정 검토 작업 파이프라인 개설이 촉구 요구 됩니다.
    
    # 그에 따라 물리적 파일이 우선 로컬 구역 장치 내부에 존재할 수 있도록 일단 선행 파싱 버킷 다운로드 수행부터 우직하게 먼저 최우선 순위로 조치합니다.
    downloaded_stems = sync_refined_data(dataset_dir) # 이 트리거가 발동하여 클라우드 S3 구역 발 완전 신규 파일들의 로컬 저장장치 디렉토리 내부로 다운로드를 싹 쓸어 버리며 시행 완료짓습니다.
    
    # 이 시점에서 물리적 로컬 디스크 맹목 전수 조사(SCAN) 조회를 이룩하여 진짜 사용가능한게 뭐가 있는지를 낱낱이 파악합니다.
    images_dir = Path(dataset_dir) / "images"
    all_stems = {f.stem for f in images_dir.rglob("*.jpg")}
    
    # 이미 기존에 낡은 구형 재고 문서인 옛날 train.txt 혹은 val.txt 문양 내에 기재되어져 박혀서 고착화 되어있던 잔재물 명단 현황 파이프라인 추출 조사 수행
    # 과거 이력 목록이 각인된 txt 텍스트 기재 서식을 뒤지고 패스 분석을 통해 오로지 고유한 stem 이름 파라미터 값만을 갈취 획득 추출 하여 내야만 합니다.
    existing_paths = load_existing_splits(dataset_dir)
    existing_stems = {Path(p).stem for p in existing_paths}
    
    # 진정 순수하게 새로운(Trusly new) 대상 신규 데이터 집합 세트를 최종 도출 결론 판결 식별합니다 (로컬 디스크에 실종재 존재는 하되, 기존 구형 분할 메타 현황목록에는 기명 적시 서술 존재하지 않는 항목들끼리의 빼기 연산 집합 처리 결과)
    new_stems = list(all_stems - existing_stems)
    
    print(f"🔎 동기화 현장 물리적 로컬 디스크 탐사 수사 보고 현황: 로컬 내부 전체 이미지 보유 수량 총 {len(all_stems)}건 적발, 이 중 이미 시스템에서 편입 결속 등록 절차를 밟은 기 인가 사용 이력 파일들 수량 {len(existing_stems)}건 규명 확인.")
    print(f"✨ 뺀 값을 산정하여 미 배정 순수 신규 자원 객체 파일 물량 잉여분 개체를 총 {len(new_stems)} 건 규합 색출 산출에 성공했습니다.")

    # 2. 임시 런타임 1회성 소모 재학습 전용 파생 아티팩트 파일 시스템 환경 문서 구조물(Transient Files)들을 조립 기획 제조합니다.
    yaml_path = create_transient_datasets(dataset_dir, new_stems)
    
    # 이 연산의 노고 끝에 도출된 귀하디 귀한 통합 메타데이터 yaml 경로 결과 산출 값을 향후 Airflow 관제 시스템 데몬 프로세스의 XCom 파이프나 stdout 시스템 라인을 통해 외부 전파 포집 및 체포 할 수 있게끔 전위 출력 시스템 콘솔 로그 규격으로 배출시킵니다.
    print(f"::set-output name=data_yaml::{yaml_path}")

if __name__ == "__main__":
    main()
