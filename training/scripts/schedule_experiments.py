import os
import yaml
import subprocess
import time
from itertools import product
import sys

# 예약 실행을 위한 기본 설계 환경설정 구조(Configuration) 정립
MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov11n', 'yolov11s', 'yolov11m']
IMG_SIZES = [640, 960, 1280, 1600]
BASE_CONFIG_PATH = '../configs/config.yaml'

def main():
    # config.yaml 및 구동 엔진의 run_exp.py 상주 파일들의 접근 가시성을 부여하기 위한 진입 및 스크립트 실행 디렉토리 폴더 최적 경로 동기화 절차
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 1. 뼈대가 될 기초 원본 야믈 베이스 기본 설정(Base Config) 문서 파싱 인출
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"오류 판정: 현재 실행 프로세스 경로위치인 {os.getcwd()} 공간 안에 {BASE_CONFIG_PATH} 파일이 없어 파싱할 수 없습니다.")
        return

    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # 2. 전수 조사형 모델 하이퍼파라미터 반복 순환 조합 순회(Iteration) 처리
    total_exps = len(MODELS) * len(IMG_SIZES)
    current_idx = 0

    print(f"전체 가용 판별 모델 개수 {len(MODELS)}개: {MODELS}")
    print(f"발견된 연산 이미지 화소(해상도) 배열 {len(IMG_SIZES)}개: {IMG_SIZES}")
    print(f"전체 산출되어 배정한 강제 연속 실험 작업 계획 총괄수(Total Scheduled Experiments): {total_exps}")
    print("=" * 60)
    
    for model, img_size in product(MODELS, IMG_SIZES):
        current_idx += 1
        exp_name = f"{model}_{img_size}"
        
        print(f"\n[현재 진척도: {current_idx}/{total_exps}] 모델 훈련 스케줄 실험 개시: {exp_name} (모델링 타입: {model}, 이미지 픽셀 해상도 규격: {img_size})")
        print("-" * 60)
        
        # 훈련마다 딥러닝 런타임 구조 Config 설정 임시 대체(Modify Config Overriding)
        exp_config = base_config.copy()
        exp_config['model_module'] = model
        exp_config['img_size'] = img_size
        exp_config['exp_name'] = exp_name
        
        # 복붙 후 일회적 소비 사용을 앞둔 더미 야믈 (Temp Config Name) 이름 명명 부여
        temp_config_path = f"config_{exp_name}.yaml"
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(exp_config, f)
            
        try:
            # 병렬 또는 백그라운드 구동 실험 진수 (Run Experiment Engine)
            # 실행 환경 안정성을 보장하기 위해 구동기 커맨드의 python 실행 체계를 현행 (sys.executable)로 지목 반영 강제화합니다.
            cmd = [sys.executable, "run_exp.py", "--config", temp_config_path]
            
            start_t = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_t
            
            print(f"✅ 예정된 {exp_name} 실험 세트가 정상적인 시스템 완주를 맺었습니다. 소요시간은 대략 {duration/60:.1f} 분(Mins)이 걸려 경과됨을 추적 확인했습니다.")
            
        except subprocess.CalledProcessError:
            print(f"❌ 분석기법 훈련 세션 도중 알 수 없는 충돌 결함으로 실험 객체인 {exp_name} 세트 구동이 비정상 종료(Failed) 당했습니다. 오류 증상을 진단하기위해 런타임 로그 파일을 필히 정밀 타격하여 주시길 권장합니다. 곧바로 스케줄 된 다음 타겟 실행 루프로 점프 이행중...")
        except KeyboardInterrupt:
            print("\n⚠️ 콘솔 사용자 또는 서버 지휘 통제기(Operator)에 의해 강건 제어 인터럽트가 유발 되어 진행하던 복수 실험 스케줄 구동 과정 일체 전체가 완전히 박탈 되었습니다. (정지). 현지 점유 파라미터 임시 저장 파일(Temp) 등의 삭제 등을 수행한 연후 최종 프로세스 종료시킴...")
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            sys.exit(1)
        finally:
            # 런타임 수명 주기가 종료된 후 폐기처분 (Cleanup) 단계의 더미 가상 컨픽 파기 및 흔적 삭제
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    print("\n" + "=" * 60)
    print("🎉 예약되어 할당된 모든 종합 벤치마크 실험 스케줄링 운영(Scheduler Queue Array) 일정이 완전하게 끝맺음을 장식했습니다.")

if __name__ == "__main__":
    main()
