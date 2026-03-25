
import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import importlib.util

# 1. plate_engine_pro.py 동적 로드
sys.path.insert(0, str(Path(__file__).parent))
engine_path = Path(__file__).parent / "plate_engine_pro.py"
spec = importlib.util.spec_from_file_location("plate_engine_pro", engine_path)
plate_engine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plate_engine_module)
PlateEnginePro = plate_engine_module.PlateEnginePro

class RealTimePlateRecognizer:
    def __init__(self, yolo_model='best.pt'):
        print("=" * 50)
        print("  실시간 고정확도 번호판 인식기 (Pro)")
        print("=" * 50)
        
        # 엔진 초기화
        self.engine = PlateEnginePro()
        
        # 고정확도 설정을 위한 임계값 강제 조정
        self.engine.config.DETECT_CONF = 0.45
        self.engine.config.OCR_CONF = 0.65
        self.engine.consecutive_required = 3  # 3프레임 연속 일치 시에만 인정
        
        self.detected_plates = defaultdict(list)
        print("[정보] 엔진 최적화 완료 (정확도 우선 모드)")

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[오류] 영상을 열 수 없습니다: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 윈도우 생성
        cv2.namedWindow("Real-Time ANPR Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-Time ANPR Pro", 1280, 720)

        print(f"[영상] {video_path} 처리 시작...")
        print("[안내] Q: 종료, Space: 일시정지")

        frame_idx = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                # 번호판 인식 (엔진 내부에서 스킵/캐싱 처리됨)
                results = self.engine.process_frame(frame)
                
                # 결과 저장 및 화면 그리기
                display_frame = frame.copy()
                for r in results:
                    plate = r['plate']
                    conf = r['confidence']
                    bbox = list(map(int, r['bbox']))
                    
                    # 99% 정확도를 목표로 하므로, 통계 데이터에 추가
                    self.detected_plates[plate].append(conf)
                    
                    # 박스 그리기
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # 텍스트 표시 (한글 폰트 문제로 영문/숫자 위주 표시, 엔진 내부 draw_korean_text 활용 가능)
                    label = f"{plate} ({conf:.2f})"
                    cv2.putText(display_frame, label, (bbox[0], bbox[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 하단 정보 바
                info_text = f"Frame: {frame_idx} | Plates: {len(self.detected_plates)}"
                cv2.putText(display_frame, info_text, (20, height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Real-Time ANPR Pro", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()
        self.print_summary()

    def print_summary(self):
        print("\n" + "=" * 50)
        print("  최종 인식 결과 (고정확도 검증)")
        print("=" * 50)
        
        if not self.detected_plates:
            print("인식된 데이터가 없습니다.")
            return

        # 평균 정확도가 높은 순으로 정렬
        summary = []
        for plate, confs in self.detected_plates.items():
            avg_conf = np.mean(confs)
            count = len(confs)
            # 신뢰도가 낮은 단발성 인식은 제외 (정확도 99% 목표를 위해)
            if count >= 3: 
                summary.append((plate, avg_conf, count))

        summary.sort(key=lambda x: x[1], reverse=True)

        print(f"{'번호판':<15} | {'평균 정확도':<10} | {'인식 횟수':<10}")
        print("-" * 50)
        for plate, avg, cnt in summary:
            print(f"{plate:<15} | {avg:.2%} | {cnt:<10}")
        
        print(f"\n[결론] 총 {len(summary)}개의 고신뢰 번호판을 확정했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="동영상 파일 경로")
    args = parser.parse_args()
    
    recognizer = RealTimePlateRecognizer()
    recognizer.process(args.video)
