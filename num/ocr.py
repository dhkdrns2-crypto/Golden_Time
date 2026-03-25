import sys
import json
import cv2
import re
from collections import Counter
from paddleocr import PaddleOCR

def clean_plate_number(text):
    # 공백 제거 및 한글/숫자만 남기기
    cleaned = re.sub(r'[^0-9가-힣]', '', text)
    
    # 한국 번호판은 보통 숫자로 시작하고 한글이 포함되어야 함
    # 최소 5자 이상, 한글 1자 이상 포함 조건 추가
    if len(cleaned) >= 5 and re.search(r'[가-힣]', cleaned):
        return cleaned
    return None

# OCR 엔진 전역 초기화 (속도 향상 및 메모리 절약)
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR] Initializing PaddleOCR engine... (This may take a while on first run)")
        try:
            # show_log=True로 설정하여 내부 로딩 과정을 볼 수 있게 함
            _ocr_engine = PaddleOCR(lang='korean', use_angle_cls=True, show_log=True)
            print("[OCR] PaddleOCR engine initialized successfully.")
        except Exception as e:
            print(f"[OCR] Failed to initialize engine: {str(e)}")
            raise e
    return _ocr_engine

def recognize_plate_from_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[OCR] Processing video: {total_frames} frames, {fps} FPS")

        ocr = get_ocr_engine()
        detected_plates = []
        confidences = []
        
        frame_count = 0
        processed_count = 0
        
        # 성능을 위해 1초에 한 번꼴(30프레임 간격)로 분석
        skip_interval = int(fps)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % skip_interval == 0:
                processed_count += 1
                print(f"[OCR] Analyzing frame {frame_count}/{total_frames}...", end='\r')
                
                result = ocr.ocr(frame)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        cleaned = clean_plate_number(text)
                        if cleaned:
                            print(f"\n[OCR] Detected candidate: {cleaned} ({conf:.2f})")
                            detected_plates.append(cleaned)
                            confidences.append(conf)
                
                # 충분한 샘플(예: 10개 이상)이 모이면 조기 종료하여 시간 단축
                if len(detected_plates) >= 10:
                    print(f"\n[OCR] Collected enough samples ({len(detected_plates)}), finishing early.")
                    break

            frame_count += 1

        cap.release()
        print(f"\n[OCR] Analysis finished. Total frames processed: {processed_count}")

        if not detected_plates:
            return {"error": "No plate detected in the video"}

        # 가장 많이 등장한 번호판 선택 (Voting)
        plate_counts = Counter(detected_plates)
        most_common_plate = plate_counts.most_common(1)[0][0]
        
        # 해당 번호판의 평균 신뢰도 계산
        avg_confidence = sum([c for p, c in zip(detected_plates, confidences) if p == most_common_plate]) / plate_counts[most_common_plate]

        return {
            "detected_plate": most_common_plate, 
            "confidence": avg_confidence,
            "total_frames_processed": frame_count // 5,
            "detection_count": plate_counts[most_common_plate]
        }
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file_path = sys.argv[1]
        ocr_result = recognize_plate_from_video(video_file_path)
        print(json.dumps(ocr_result, ensure_ascii=False))
    else:
        print(json.dumps({"error": "No video file path provided"}, ensure_ascii=False))
