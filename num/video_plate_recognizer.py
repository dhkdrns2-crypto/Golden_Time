import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import os
import time
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from ultralytics import YOLO

import importlib.util

# plate_engine_pro.py 파일 경로를 직접 지정
engine_path = Path(__file__).parent / "plate_engine_pro.py"

if not engine_path.exists():
    print(f"[오류] {engine_path}를 찾을 수 없습니다. 스크립트와 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

try:
    # 파일을 모듈로 로드
    spec = importlib.util.spec_from_file_location("plate_engine_pro", engine_path)
    plate_engine_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plate_engine_module)
    PlateEnginePro = plate_engine_module.PlateEnginePro
    print("[정보] plate_engine_pro.py를 성공적으로 로드했습니다.")
except Exception as e:
    print(f"[오류] plate_engine_pro.py 로드 중 오류 발생: {e}")
    sys.exit(1)

class VideoPlateProcessor:
    def __init__(self, yolo_model_path='best.pt', ocr_model_path='plate_ocr_crnn.pth'):
        print("=" * 50)
        print("  동영상 번호판 일괄 인식기")
        print("=" * 50)
        
        self.engine = PlateEnginePro()
        print("[정보] PlateEnginePro 초기화 완료.")

        try:
            print(f"[정보] YOLO 모델을 '{yolo_model_path}'로 교체합니다.")
            self.engine.model = YOLO(yolo_model_path)
            print(f"[정보] OCR 모델은 PlateEnginePro의 기본 설정('{ocr_model_path}')을 사용합니다.")
            print("[준비 완료]\n")
        except Exception as e:
            print(f"[오류] 모델 교체 중 오류 발생: {e}")
            print("기본 모델로 계속 진행합니다.")

        self.detected_plates = defaultdict(list)

    def process_video(self, video_path: str, frame_skip: int = 5):
        """
        Process a video file to detect and recognize license plates.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[오류] 영상을 열 수 없습니다: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[영상 정보] 경로: {video_path}")
        print(f"[영상 정보] 총 프레임: {total_frames}, FPS: {fps:.2f}")
        print(f"[처리 설정] {frame_skip+1} 프레임마다 1번씩 처리합니다.")
        print("-" * 50)

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (frame_skip + 1) == 0:
                results = self.engine.process_frame(frame)
                
                for r in results:
                    plate_text = r.get('text', r.get('plate', ''))
                    confidence = r.get('ocr_confidence', r.get('confidence', 0))
                    
                    if plate_text and len(plate_text) >= 7:
                        self.detected_plates[plate_text].append(confidence)
                
                progress = (frame_count / total_frames) * 100
                sys.stdout.write(f"\r[진행률] {progress:.1f}% ({frame_count}/{total_frames}) | 찾은 번호판: {len(self.detected_plates)}개")
                sys.stdout.flush()

            frame_count += 1

        cap.release()
        end_time = time.time()
        print(f"\n\n[처리 완료] 총 처리 시간: {end_time - start_time:.2f}초")
        print("-" * 50)

    def print_summary(self):
        """
        Prints a summary of the detected license plates and their average confidence.
        """
        print("\n" + "=" * 50)
        print("  인식 결과 요약")
        print("=" * 50)

        if not self.detected_plates:
            print("인식된 번호판이 없습니다.")
            return

        sorted_plates = sorted(self.detected_plates.items(), key=lambda item: len(item[1]), reverse=True)

        print(f"{'번호판':<15} | {'인식 횟수':<10} | {'평균 정확도':<15}")
        print("-" * 50)

        for plate_text, confidences in sorted_plates:
            count = len(confidences)
            avg_conf = np.mean(confidences) if confidences else 0
            print(f"{plate_text:<15} | {count:<10} | {avg_conf:.2%}")
        
        print("-" * 50)
        print(f"총 {len(sorted_plates)}개의 고유한 번호판을 인식했습니다.")


def main():
    parser = argparse.ArgumentParser(description="동영상에서 번호판을 인식하고 결과를 요약합니다.")
    parser.add_argument("video_path", help="처리할 동영상 파일의 경로. 'video' 폴더 전체를 처리하려면 'video'를 입력하세요.")
    parser.add_argument("--yolo_model", default="best.pt", help="YOLO 감지 모델 파일 경로")
    parser.add_argument("--ocr_model", default="plate_ocr_crnn.pth", help="OCR 인식 모델 파일 경로")
    parser.add_argument("--skip", type=int, default=5, help="성능을 위해 건너뛸 프레임 수")
    args = parser.parse_args()

    processor = VideoPlateProcessor(yolo_model_path=args.yolo_model, ocr_model_path=args.ocr_model)

    if os.path.isdir(args.video_path):
        video_files = [f for f in os.listdir(args.video_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            print(f"[오류] '{args.video_path}' 디렉토리에서 비디오 파일을 찾을 수 없습니다.")
            return
        print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다.")
        for video_file in video_files:
            video_path = os.path.join(args.video_path, video_file)
            processor.process_video(video_path, frame_skip=args.skip)
    elif os.path.isfile(args.video_path):
        processor.process_video(args.video_path, frame_skip=args.skip)
    else:
        print(f"[오류] 파일 또는 디렉토리를 찾을 수 없습니다: {args.video_path}")
        return

    processor.print_summary()


if __name__ == "__main__":
    main()
