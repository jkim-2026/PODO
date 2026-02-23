import yaml
import argparse
import sys
import os

# [경로 수정] src 모듈을 임포트할 수 있도록 sys.path에 프로젝트 루트를 추가합니다.
# 스크립트 위치: training/scripts/run_exp.py
# 프로젝트 루트: training/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.train import PCBTrainer
from src.inference import InferenceMgr
import importlib
from pathlib import Path 

def main():
    parser = argparse.ArgumentParser(description="PCB 결함 탐지 실험 실행기")
    # 기본 설정 파일 경로는 이제 ../configs/config.yaml 입니다.
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='설정 파일 경로')
    parser.add_argument('--data', type=str, default=None, help='재학습용 데이터 경로 오버라이드 (data.yaml)')
    parser.add_argument('--name', type=str, default=None, help='실험 이름 (예: run_id)')
    args = parser.parse_args()

    # 설정 파싱
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback: 스크립트 디렉토리를 기준으로 상대 경로를 찾습니다.
        # 사용자가 프로젝트 루트에서 'python training/run_exp.py'를 실행할 때 유용합니다.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_path = os.path.join(script_dir, config_path)
        if os.path.exists(candidate_path):
            print(f"현재 폴더에서 설정 파일 '{config_path}'을(를) 찾을 수 없습니다. '{candidate_path}'에서 로드합니다.")
            config_path = candidate_path
            
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 커맨드라인 매개변수 우선 적용 (Config Override)
    if getattr(args, 'data', None):
        print(f"🔄 데이터 경로 덮어쓰기: {args.data}")
        config['data_path'] = args.data
    if getattr(args, 'name', None):
        print(f"🔄 실험명 덮어쓰기: {args.name}")
        config['exp_name'] = args.name

    print(f"{args.config}에서 설정을 성공적으로 로드했습니다.")

    # --- 경로 해석 (상대 경로 -> 절대 경로) ---
    # 'data_path'를 프로젝트 루트(training/)를 기준으로 해석합니다.
    if 'data_path' in config and not os.path.isabs(config['data_path']):
        resolved_path = os.path.abspath(os.path.join(project_root, config['data_path']))
        print(f"상대 경로인 data_path 해석: '{config['data_path']}' -> '{resolved_path}'")
        config['data_path'] = resolved_path
    
    # --- 랜덤 시드 고정 ---
    from src.utils import set_seed, setup_logging, increment_path
    set_seed(config.get('seed', 42))
    print(f"랜덤 시드 고정: {config.get('seed', 42)}")
    
    # --- Ultralytics 설정 튜닝 ---
    # 프로젝트 루트에 위치한 'pretrained_weights' 폴더로 'weights_dir'을 가리킵니다.
    from ultralytics import settings as ul_settings
    
    weights_dir = os.path.abspath(os.path.join(project_root, "pretrained_weights"))
    
    ul_settings.update({'weights_dir': weights_dir})
    print(f"Ultralytics weights_dir 경로 설정됨: {weights_dir}")
    
    # --- 동적 모듈 로딩 (Dynamic Loading) ---
    
    # 1. 데이터셋 모듈 선택
    ds_mod_name = config.get('dataset_module', 'dataset')
    print(f"Loading Dataset Module: src/datasets/{ds_mod_name}.py")
    try:
        ds_lib = importlib.import_module(f"src.datasets.{ds_mod_name}")
        dataset = ds_lib.get_dataset(config)
    except ImportError as e:
        print(f"데이터셋 모듈 로딩 오류: {e}")
        return
    except AttributeError:
        print(f"src.datasets.{ds_mod_name} 모듈은 반드시 'get_dataset(config)' 함수가 있어야 합니다.")
        return

    # 데이터 준비
    # 사용자가 직접 구체적인 yaml 설정(Airflow sync_data 등)을 전달했다면 파싱을 건너뜁니다.
    if config['data_path'].endswith('.yaml') and os.path.exists(config['data_path']):
        print(f"데이터셋 준비 과정을 생략하고 기존 설정 파일을 사용합니다: {config['data_path']}")
        data_yaml = config['data_path']
    else:
        data_yaml = dataset.prepare()

    # 2. 모델 모듈 선택
    md_mod_name = config.get('model_module', 'yolov8')
    print(f"모델 모듈 로딩 중: src/models/{md_mod_name}.py")
    
    try:
        md_lib = importlib.import_module(f"src.models.{md_mod_name}")
        # 전체 config를 모델 팩토리(Factory)로 전달합니다.
        model = md_lib.get_model(config)
    except ImportError as e:
        print(f"모델 모듈 로딩 오류: {e}")
        return
    except AttributeError:
        print(f"src.models.{md_mod_name} 모듈은 반드시 'get_model(config)' 함수를 가지고 있어야 합니다.")
        return

    # 3. 학습(Training) 시작
    print(f"\n{'='*20} 학습 시작 (실험명: {config['exp_name']}) {'='*20}")
    
    print(f"랜덤 시드 고정: {config.get('seed', 42)}")
    

    print(f"사용 중인 데이터: {data_yaml}")
    
    trainer = PCBTrainer(model, config)
    
    # 저장 경로(save_dir) 설정 (config에 지정된 'project' 사용, 없다면 project_root 기준의 'runs' 디렉토리 사용)
    # 반드시 Project Root 안의 'runs'에 저장되도록 강제합니다.
    runs_dir = os.path.join(project_root, "runs")
    config['project'] = runs_dir
    print(f"모델 저장 디렉토리 설정: {runs_dir}")
    
    # --- 수동 디렉토리 생성 및 로깅 셋업 ---
    # 로그를 남길 위치를 지정하기 위해 학습이 시작되기 *전에* 대상 폴더를 확정해야 합니다.
    # 이를 위해 'increment_path' 로직을 수동으로 호출하여 폴더명을 결정합니다.
    
    initial_exp_name = config['exp_name']
    base_save_dir = os.path.join(runs_dir, initial_exp_name)
    
    # 사용 비어있는 폴더 번호를 찾습니다 (예: runs/baseline -> runs/baseline2)
    save_dir = increment_path(Path(base_save_dir), exist_ok=False, mkdir=True)
    save_dir = str(save_dir) # 문자열로 치환
    
    # YOLO 학습기가 이 폴더를 정확히 쓰도록 config를 실제 폴더명(예:'baseline2')으로 업데이트합니다.
    # 전체 경로에서 마지막 폴더명만 추출합니다.
    actual_exp_name = os.path.basename(save_dir)
    config['exp_name'] = actual_exp_name
    
    # 'runs/exp_name/console.log'에 로깅하도록 설정
    setup_logging(save_dir)
    
    try:
        # 학습 진행
        actual_save_dir = trainer.train(data_yaml)
        if actual_save_dir:
            save_dir = str(actual_save_dir)

        print("\n학습이 성공적으로 완료되었습니다.")

        # 3. 테스트(Test) 데이터셋을 사용한 최종 추론 실행
        # 로드될 모델의 경로는 보통 {save_dir}/weights/best.pt 입니다.
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        
        print(f"다음 모델을 사용하여 테스트 세트(Test set) 推論(Inference)를 진행합니다: {best_model_path}")
        
        if os.path.exists(best_model_path):
            inference_mgr = InferenceMgr(best_model_path, config)
            test_list_path = os.path.join(config['data_path'], 'test_images.txt')
            
            if os.path.exists(test_list_path):
                # {save_dir}/inference 저장
                inference_mgr.predict(test_list_path, project=save_dir, name="inference")
        
        else:
            print(f"최고 성능 모델 파일이 보이지 않습니다: {best_model_path}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 실험이 강제 종료되었습니다.")
    except Exception as e:
        print(f"\n실험 실행 중 예기치 않은 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # --- 학습 완료 후의 구조화/정리(Cleanup) 작업을 수행합니다. ---
        # 실제 로직은 src/utils.py 에 위임.
        from src.utils import cleanup_artifacts
        cleanup_artifacts(save_dir, config)

if __name__ == "__main__":
    main()
