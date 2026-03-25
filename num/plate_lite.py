"""
YOLO26 Lite — 한국 차량 번호판 인식 경량 CLI 버전
GUI 없이 터미널에서 실행. plate_engine_pro.py 엔진 활용.

사용법:
    python plate_lite.py movie/hiway.mp4      # 영상 인식
    python plate_lite.py image.jpg             # 이미지 인식
    python plate_lite.py 22/                   # 폴더 내 이미지 일괄 인식
"""

import sys
import os
import re
import time
import cv2
import numpy as np
from pathlib import Path

# 엔진 임포트
sys.path.insert(0, str(Path(__file__).parent))
from plate_engine_pro import PlateEnginePro


import requests

class PlateLiteCLI:
    """경량 CLI 번호판 인식"""

    def __init__(self, backend_url="http://localhost:1111/api/ocr/save"):
        print("=" * 50)
        print("  YOLO26 Lite — 한국 번호판 인식 CLI")
        print("=" * 50)
        self.engine = PlateEnginePro()
        self._seen = {}
        self.backend_url = backend_url
        print("[준비 완료]\n")

    def send_to_backend(self, gt_id, plate, confidence, is_final=True):
        """백엔드로 OCR 결과 전송"""
        if not gt_id:
            return
        try:
            data = {
                "gtId": int(gt_id),
                "detectedPlate": plate,
                "confidence": float(confidence),
                "isFinal": is_final
            }
            response = requests.post(self.backend_url, json=data, timeout=2)
            if response.status_code == 200:
                print(f"[API] 전송 성공: {plate} (gtId={gt_id})")
            else:
                print(f"[API] 전송 실패: {response.status_code} {response.text}")
        except Exception as e:
            print(f"[API] 오류: {e}")

    def _reset_tracker(self):
        """트래킹/캐시 완전 초기화 (정적 이미지용)"""
        self.engine.recent_plates.clear()
        self.engine._ocr_track_cache.clear()
        if hasattr(self.engine, '_global_plate_history'):
            self.engine._global_plate_history.clear()
        if hasattr(self.engine, '_frame_counter'):
            self.engine._frame_counter = 0
        if hasattr(self.engine, 'plate_tracker'):
            self.engine.plate_tracker = {}

    def process_image(self, image: np.ndarray, reset: bool = True) -> list[dict]:
        """단일 이미지 인식 (reset=True면 트래커 초기화)"""
        if reset:
            self._reset_tracker()
        results = self.engine.process_frame(image)
        out = []
        for r in results:
            text = r.get('text', r.get('plate', ''))
            conf = r.get('ocr_confidence', r.get('confidence', 0))
            if text and len(text) >= 5:
                out.append({'plate': text, 'confidence': conf, 'bbox': r.get('bbox', [])})
                self._seen[text] = self._seen.get(text, 0) + 1
        return out

    def _draw_text(self, frame, text, pos, color=(0, 255, 0), size=28):
        """한글 텍스트를 프레임 위에 그리기 (PIL 사용)"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            try:
                font = ImageFont.truetype("malgun.ttf", size)
            except Exception:
                font = ImageFont.load_default()
            # 배경 박스
            x, y = int(pos[0]), int(pos[1])
            tw = len(text) * size
            cv2.rectangle(frame, (x, y - size - 5), (x + tw, y), (0, 0, 0), -1)
            # PIL로 한글 렌더링
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((x, y - size - 3), text, font=font, fill=(color[2], color[1], color[0]))
            frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            # PIL 없으면 cv2 기본 (한글 깨짐)
            cv2.putText(frame, text, (int(pos[0]), int(pos[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def process_video(self, video_path: str, skip_frames: int = 2, gt_id: int = None):
        """영상 인식 + 화면 표시 (OCR 백그라운드 스레드로 FPS 유지)"""
        import threading, queue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[오류] 영상 열기 실패: {video_path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))
        print(f"[영상] {video_path} ({total}프레임, {fps:.1f}FPS)")
        if gt_id:
            print(f"[연동] gtId={gt_id}")
        print("[조작] Q=종료, Space=일시정지\n")

        # 공유 상태
        last_results = []
        last_result_time = time.time()
        result_lock = threading.Lock()

        ocr_queue = queue.Queue(maxsize=1)
        det_count = 0
        running = True
        RESULT_TTL = 0.5  # 결과 유지 시간 (초) — 잔상 방지용 단축

        # ── 실시간 YOLO 번호판 탐지 + OCR ──
        live_boxes = []  # 현재 프레임 번호판 bbox
        live_lock = threading.Lock()

        # ── OCR 워커: 라이브 bbox 크롭에서 직접 OCR ──
        def ocr_worker():
            nonlocal last_results, last_result_time, det_count
            while running:
                try:
                    fid, frm, boxes = ocr_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if not boxes:
                    with result_lock:
                        last_results = []
                    continue
                t0 = time.time()
                results = []
                # 전체 파이프라인으로 OCR (정확도 우선)
                fb_results = self.process_image(frm, reset=False)
                if fb_results:
                    results = fb_results
                elapsed = time.time() - t0
                with result_lock:
                    if results:
                        last_results = results
                        last_result_time = time.time()
                        # 백엔드 전송 (gt_id가 있을 때만)
                        if gt_id:
                            for r in results:
                                self.send_to_backend(gt_id, r['plate'], r['confidence'])
                    else:
                        # 전체 파이프라인 폴백 (bbox OCR 실패 시)
                        fb_results = self.process_image(frm, reset=False)
                        if fb_results:
                            last_results = fb_results
                            last_result_time = time.time()
                        else:
                            last_results = []
                for r in results:
                    det_count += 1
                    self._seen[r['plate']] = self._seen.get(r['plate'], 0) + 1
                    cnt = self._seen[r['plate']]
                    print(f"[F:{fid:>5}/{total}] {r['plate']:<15} "
                          f"conf={r['confidence']:.0%}  x{cnt}  ({elapsed:.1f}s)")

        worker = threading.Thread(target=ocr_worker, daemon=True)
        worker.start()

        frame_idx = 0
        paused = False
        det_interval = max(int(fps // 5), 2)   # 초당 5회 YOLO+OCR

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
            else:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = False
                continue

            # ── YOLO 번호판 탐지 + OCR 큐 ──
            if frame_idx % det_interval == 0:
                try:
                    dets = self.engine.model(frame, conf=0.25, imgsz=640, verbose=False)
                    boxes = []
                    for det in dets[0].boxes:
                        bx1, by1, bx2, by2 = map(int, det.xyxy[0].tolist())
                        bconf = float(det.conf[0])
                        if (bx2 - bx1) >= 20 and (by2 - by1) >= 8:
                            boxes.append([bx1, by1, bx2, by2, bconf])
                    with live_lock:
                        live_boxes = boxes
                    # OCR 큐에 프레임 + bbox 전달
                    if ocr_queue.empty():
                        try:
                            ocr_queue.put_nowait((frame_idx, frame.copy(), list(boxes)))
                        except queue.Full:
                            pass
                except Exception:
                    pass

            # ── 화면 그리기 ──
            display = frame.copy()

            # 1) 실시간 YOLO bbox (노란색 — 인식 대기)
            with live_lock:
                cur_live = list(live_boxes)
            # 2) OCR 결과 (초록색 — 인식 완료)
            with result_lock:
                if time.time() - last_result_time > RESULT_TTL:
                    last_results = []
                current_results = list(last_results)

            # ── YOLO bbox: 노란색으로 번호판 탐지 위치 표시 ──
            for bx1, by1, bx2, by2, bconf in cur_live:
                bh = by2 - by1
                if current_results:
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                else:
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

            # ── OCR 결과: 화면 하단 HUD에 고정 표시 ──
            h_disp, w_disp = display.shape[:2]
            if current_results:
                for ri, r in enumerate(current_results):
                    label = f"{r['plate']}  {r['confidence']:.0%}"
                    # 하단 반투명 배경
                    hud_y = h_disp - 70 - ri * 50
                    cv2.rectangle(display, (10, hud_y - 5), (400, hud_y + 40), (0, 0, 0), -1)
                    self._draw_text(display, label, (15, hud_y + 35), (0, 255, 0), 32)
            elif cur_live:
                hud_y = h_disp - 70
                cv2.rectangle(display, (10, hud_y - 5), (200, hud_y + 40), (0, 0, 0), -1)
                self._draw_text(display, "인식중...", (15, hud_y + 35), (0, 255, 255), 28)

            info = f"YOLO26 Lite | F:{frame_idx}/{total} | FPS:{fps:.0f} | Det:{det_count}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            h, w = display.shape[:2]
            if w > 1280:
                scale = 1280 / w
                display = cv2.resize(display, (1280, int(h * scale)))
            cv2.imshow('YOLO26 Lite', display)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = True

        running = False
        worker.join(timeout=3)
        cap.release()
        cv2.destroyAllWindows()
        self._print_summary(det_count)

    def process_folder(self, folder_path: str):
        """폴더 일괄 인식"""
        folder = Path(folder_path)
        images = sorted(folder.glob('*.png')) + sorted(folder.glob('*.jpg'))
        if not images:
            print(f"[오류] 이미지 없음: {folder_path}")
            return

        print(f"[폴더] {folder_path} ({len(images)}개)\n")
        correct = 0

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                buf = np.fromfile(str(img_path), dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is None:
                continue

            t0 = time.time()
            results = self.process_image(image)
            elapsed = time.time() - t0

            gt = re.sub(r'\.(png|jpg)$', '', img_path.stem, flags=re.IGNORECASE)
            gt = re.sub(r'^(트럭\s*|버스\s*)', '', gt).strip()

            if results:
                best = max(results, key=lambda x: x['confidence'])
                ok = best['plate'] == gt
                if ok:
                    correct += 1
                print(f"  {'✅' if ok else '❌'} {img_path.name:<25} {gt:<12} → {best['plate']:<12} "
                      f"conf={best['confidence']:.0%}  ({elapsed:.1f}s)")
            else:
                print(f"  ❌ {img_path.name:<25} {gt:<12} → (실패)  ({elapsed:.1f}s)")

        print(f"\n  정확도: {correct}/{len(images)} = {correct/max(len(images),1)*100:.1f}%")

    def _print_summary(self, det_count):
        print("\n" + "=" * 50)
        print("  인식 결과 요약")
        print("=" * 50)
        for plate, cnt in sorted(self._seen.items(), key=lambda x: x[1], reverse=True):
            print(f"  {plate:<15} x{cnt}")
        print(f"\n  총 {len(self._seen)}개 번호판, {det_count}회 감지")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    target = Path(sys.argv[1])
    gt_id = sys.argv[2] if len(sys.argv) > 2 else None
    cli = PlateLiteCLI()

    if target.is_dir():
        cli.process_folder(str(target))
    elif target.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov'):
        cli.process_video(str(target), gt_id=gt_id)
    elif target.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
        image = cv2.imread(str(target))
        if image is None:
            buf = np.fromfile(str(target), dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        results = cli.process_image(image)
        for r in results:
            print(f"  {r['plate']}  conf={r['confidence']:.0%}")
            if gt_id:
                cli.send_to_backend(gt_id, r['plate'], r['confidence'])
        if not results:
            print("  (인식 실패)")
    else:
        print(f"[오류] 지원 안 됨: {target}")


if __name__ == '__main__':
    main()
