
import sys
import os
import time
import cv2
import numpy as np
import re
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import threading
import queue
from PIL import Image, ImageDraw, ImageFont
import importlib.util

# --- 한글 텍스트 표시 최적화 (Pillow 캐시 기반) ---
_kr_font_cache = {}
_kr_text_cache = {}

def get_korean_text_overlay(text, color=(0, 255, 0), size=24):
    cache_key = (text, color, size)
    if cache_key in _kr_text_cache:
        return _kr_text_cache[cache_key]
    
    if size not in _kr_font_cache:
        try:
            _kr_font_cache[size] = ImageFont.truetype("malgun.ttf", size)
        except:
            _kr_font_cache[size] = ImageFont.load_default()
    
    font = _kr_font_cache[size]
    try:
        left, top, right, bottom = font.getbbox(text)
        tw, th = right - left, bottom - top
    except:
        tw, th = len(text) * size, size
        
    tmp = Image.new("RGBA", (tw + 10, th + 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    b, g, r = color
    draw.text((5, 5), text, font=font, fill=(r, g, b, 255))
    
    tmp_np = np.array(tmp)
    alpha = tmp_np[:, :, 3:4].astype(np.float32) / 255.0
    img_bgr = tmp_np[:, :, :3][:, :, ::-1].astype(np.float32)
    
    _kr_text_cache[cache_key] = (img_bgr, alpha)
    return img_bgr, alpha

def draw_korean_text_fast(frame, text, pos, color=(0, 255, 0), size=24):
    img_bgr, alpha = get_korean_text_overlay(text, color, size)
    x, y = int(pos[0]), int(pos[1])
    h, w = img_bgr.shape[:2]
    fh, fw = frame.shape[:2]
    
    if y < 0: y = 0
    if x < 0: x = 0
    if y + h > fh: h = fh - y
    if x + w > fw: w = fw - x
    
    if h <= 0 or w <= 0: return frame
    
    roi = frame[y:y+h, x:x+w].astype(np.float32)
    frame[y:y+h, x:x+w] = (alpha[:h, :w] * img_bgr[:h, :w] + (1 - alpha[:h, :w]) * roi).astype(np.uint8)
    return frame

# --- 엔진 동적 로드 (plate_engine_pro.py) ---
sys.path.insert(0, str(Path(__file__).parent))
try:
    engine_path = Path(__file__).parent / "plate_engine_pro.py"
    spec = importlib.util.spec_from_file_location("plate_engine_pro", engine_path)
    plate_engine_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plate_engine_module)
    PlateEnginePro = plate_engine_module.PlateEnginePro
except Exception as e:
    print(f"[오류] 엔진 로드 실패: {e}")
    sys.exit(1)

class PaddleOCRSimulator:
    def __init__(self, model_path='best.pt'):
        print("="*50)
        print("  PaddleOCR 기반 고성능 시뮬레이터 (plate.py v5.0)")
        print("="*50)
        
        # PlateEnginePro 사용 (PaddleOCR 포함)
        self.engine = PlateEnginePro()
        self.engine.config.DETECT_CONF = 0.45
        self.engine.config.OCR_CONF = 0.65
        self.engine.consecutive_required = 3
        
        self.detected_plates = defaultdict(list)
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_results = []
        self.running = True
        self.frame_count = 0
        self.display_res = (1280, 720)

    def worker_thread(self):
        """인식 전용 백그라운드 스레드"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                results = self.engine.process_frame(frame)
                self.latest_results = results
                
                # 결과 저장 (100회 제한)
                for r in results:
                    p = r['plate']
                    c = r['confidence']
                    if len(self.detected_plates[p]) < 100:
                        self.detected_plates[p].append(c)
                
            except queue.Empty: continue
            except Exception as e: print(f"Worker error: {e}")

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[오류] 영상을 열 수 없습니다: {video_path}")
            return
        
        target_fps = cap.get(cv2.CAP_PROP_FPS) or 60
        frame_delay = 1.0 / target_fps
        
        threading.Thread(target=self.worker_thread, daemon=True).start()
        
        cv2.namedWindow("ANPR Simulation", cv2.WINDOW_AUTOSIZE)
        start_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                
                self.frame_count += 1
                
                display_frame = cv2.resize(frame, self.display_res)
                scale_x = self.display_res[0] / frame.shape[1]
                scale_y = self.display_res[1] / frame.shape[0]

                if self.frame_queue.empty():
                    self.frame_queue.put_nowait(frame)

                for r in self.latest_results:
                    p = r['plate']
                    c = r['confidence']
                    bx = [int(r['bbox'][0]*scale_x), int(r['bbox'][1]*scale_y), 
                          int(r['bbox'][2]*scale_x), int(r['bbox'][3]*scale_y)]
                    
                    cv2.rectangle(display_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                    draw_korean_text_fast(display_frame, f"{p} ({c:.2f})", (bx[0], bx[1]-35), size=26)

                curr_fps = self.frame_count / (time.time() - start_time)
                info = f"FPS: {curr_fps:.1f} | Detected: {len(self.detected_plates)}"
                cv2.rectangle(display_frame, (5, 5), (300, 50), (0,0,0), -1)
                draw_korean_text_fast(display_frame, info, (15, 12), (255,255,255), 22)
                
                cv2.imshow("ANPR Simulation", display_frame)

                wait_ms = int((frame_delay - (time.time() - loop_start)) * 1000)
                if cv2.waitKey(max(1, wait_ms)) & 0xFF == ord('q'): break
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print("  PaddleOCR 기반 시뮬레이션 인식 결과 요약 (100회 제한)")
        print("="*50)
        
        summary = sorted([(p, np.mean(c), len(c)) for p, c in self.detected_plates.items() if len(c) >= 2], 
                        key=lambda x: x[1], reverse=True)
        
        print(f"{'번호판':<15} | {'평균 정확도':<10} | {'인식 횟수':<10}")
        print("-" * 50)
        for plate, avg, cnt in summary:
            print(f"{plate:<15} | {avg:.2%} | {cnt:<10}")
        
        print(f"\n[완료] 총 {len(summary)}개의 고신뢰 번호판을 감지했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs='?', default="video/car3.mp4", help="동영상 파일 경로")
    args = parser.parse_args()
    
    simulator = PaddleOCRSimulator()
    simulator.run(args.video)
