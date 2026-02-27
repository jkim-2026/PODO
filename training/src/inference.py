import os
from ultralytics import YOLO

class InferenceMgr:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.config = config
        # 출력 디렉토리는 predict() 과정에서 런타임에 동적으로 결정됩니다.

    def predict(self, source_txt, project="runs", name="inference"):
        """
        source_txt (테스트 세트)에 나열된 모든 이미지에 대하여 예측을 수행합니다.
        """
        print(f"{source_txt}에 기재된 이미지들에 대해 추론을 진행합니다...")
        
        # YOLOv8은 파일 목록(list) 자체를 인자로 받을 수 있지만, 우선 txt 파일을 파싱합니다.
        with open(source_txt, 'r') as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
            
        # 추론 연산 실행 (Inference)
        results = self.model.predict(
            source=files,
            imgsz=self.config['img_size'],
            conf=0.25, # 기본 신뢰도 임계값
            save=True,
            save_txt=True,
            project=project,
            name=name,
            exist_ok=True
        )
        
        # 출력 디렉토리 분석확인 (YOLO는 자체적으로 project/name 경로 구조를 생성합니다)
        save_dir = os.path.join(project, name)
        
        # CSV 파일 생성
        self.generate_csv(results, save_dir)
        
        return results

    def generate_csv(self, results, save_dir):
        """
        YOLO의 Results 객체들을 파싱하여 단일 CSV 파일 체계로 변환하여 저장합니다.
        포맷 형태: ImageID, Label, Confidence, xmin, ymin, xmax, ymax
        """
        import pandas as pd
        
        csv_data = []
        
        print("\nsubmission.csv 파일을 생성 중입니다...")
        for res in results:
            # 이미지 ID (파일 명칭)
            image_id = os.path.basename(res.path)
            
            # 바운딩 박스(Bounding Boxes) 처리
            boxes = res.boxes
            for i in range(len(boxes)):
                # 클래스 라벨(Class name) 추출
                cls_id = int(boxes.cls[i].item())
                label = res.names[cls_id]
                
                # 신뢰도 점수(Confidence) 추출
                conf = float(boxes.conf[i].item())
                
                # BBox 좌표(xyxy) 추출 - 이미 픽셀 단위로 환산되어 있습니다.
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                
                csv_data.append({
                    'ImageID': image_id,
                    'Label': label,
                    'Conf': conf,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2
                })
        
        if csv_data:
            os.makedirs(save_dir, exist_ok=True)
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(save_dir, 'submission.csv')
            df.to_csv(csv_path, index=False)
            print(f"CSV 저장 위치: {csv_path}")
        else:
            print("탐지된 객체가 없습니다. CSV 파일이 생성되지 않았습니다.")

    def draw_bbox(self):
        # 파라미터가 'save=True' 인 경우 YOLOv8이 자체 구조에 의해 바운딩 박스를 자동으로 드로잉합니다.
        # 본 메서드는 향후 더욱 최적화된 맞춤형 커스텀 시각화 처리가 필요해질 경우를 대비해 뼈대만 남깁니다.
        pass
