
import sys
import os

# 프로젝트 최상단 디렉토리(root)를 파이썬 패스에 주입하여 모듈들의 상호작용 및 절대경로 참조를 가능하게 합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.qat import recalibrate

if __name__ == "__main__":
    recalibrate.main()
