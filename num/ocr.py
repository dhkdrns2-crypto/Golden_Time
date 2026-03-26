import sys
import json
import cv2
import re
import inspect
from collections import Counter
from paddleocr import PaddleOCR

try:
    # 프로젝트에 이미 있는 고정확도 엔진(우선 사용)
    from plate_engine_pro import PlateEnginePro
    _HAS_PLATE_ENGINE_PRO = True
except Exception:
    PlateEnginePro = None
    _HAS_PLATE_ENGINE_PRO = False

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
_plate_engine_pro = None

def get_plate_engine_pro():
    global _plate_engine_pro
    if not _HAS_PLATE_ENGINE_PRO:
        return None
    if _plate_engine_pro is None:
        print("[OCR] Initializing PlateEnginePro... (YOLO+OCR)")
        _plate_engine_pro = PlateEnginePro()
        print("[OCR] PlateEnginePro initialized successfully.")
    return _plate_engine_pro

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR] Initializing PaddleOCR engine... (This may take a while on first run)")
        try:
            # PaddleOCR 버전마다 지원 인자가 달라 안전하게 구성
            kwargs = dict(lang='korean', use_angle_cls=True)
            sig = None
            try:
                sig = inspect.signature(PaddleOCR.__init__)
            except Exception:
                sig = None

            if sig is not None:
                params = sig.parameters
                if 'show_log' in params:
                    kwargs['show_log'] = True
                if 'use_gpu' in params:
                    kwargs['use_gpu'] = False
                if 'use_mkldnn' in params:
                    kwargs['use_mkldnn'] = False

            _ocr_engine = PaddleOCR(**kwargs)
            print("[OCR] PaddleOCR engine initialized successfully.")
        except Exception as e:
            print(f"[OCR] Failed to initialize engine: {str(e)}")
            raise e
    return _ocr_engine

def _iter_ocr_lines(result):
    """
    PaddleOCR.ocr 반환 구조가 버전/옵션에 따라 달라서 방어적으로 순회한다.
    기대하는 line 형식: [box, (text, conf)] 또는 (box, (text, conf))
    """
    if not result:
        return []
    # 케이스1: [ [line, line, ...] ]
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        return result[0]
    # 케이스2: [line, line, ...]
    if isinstance(result, list):
        return result
    return []

def recognize_plate_from_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[OCR] Processing video: {total_frames} frames, {fps} FPS")

        engine_pro = get_plate_engine_pro()
        ocr = None if engine_pro is not None else get_ocr_engine()
        detected_plates = []
        confidences = []
        
        frame_count = 0
        processed_count = 0
        
        # 성능을 위해 1초에 한 번꼴로 분석(최소 10프레임 간격)
        skip_interval = max(int(fps), 10)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % skip_interval == 0:
                processed_count += 1
                print(f"[OCR] Analyzing frame {frame_count}/{total_frames}...", end='\r')

                if engine_pro is not None:
                    # 프로젝트 기존의 고정확도 파이프라인 사용 (YOLO 탐지 + OCR + 검증/보정)
                    results = engine_pro.process_frame(frame)
                    for r in results:
                        plate = r.get("plate")
                        conf = float(r.get("confidence", 0.0))
                        if plate:
                            print(f"\n[OCR] Detected candidate: {plate} ({conf:.2f})")
                            detected_plates.append(plate)
                            confidences.append(conf)
                else:
                    # PaddleOCR-only 폴백 (정확도 낮음)
                    h, w = frame.shape[:2]
                    y1, y2 = int(h * 0.55), int(h * 0.95)
                    x1, x2 = int(w * 0.20), int(w * 0.80)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0 and (roi.shape[1] < 900):
                        scale = 900 / max(1, roi.shape[1])
                        roi = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)))

                    result = ocr.ocr(roi) if roi.size > 0 else ocr.ocr(frame)
                    lines = list(_iter_ocr_lines(result))
                    if not lines:
                        result = ocr.ocr(frame)
                        lines = list(_iter_ocr_lines(result))

                    for line in lines:
                        try:
                            payload = line[1] if isinstance(line, (list, tuple)) and len(line) >= 2 else None
                            if not payload or not isinstance(payload, (list, tuple)) or len(payload) < 2:
                                continue
                            text = payload[0]
                            conf = float(payload[1])
                        except Exception:
                            continue

                        cleaned = clean_plate_number(str(text))
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
            "total_frames_processed": processed_count,
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
