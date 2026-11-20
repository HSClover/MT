import os
import sys
import time
import json
import queue
import threading
from pathlib import Path
from collections import deque
from typing import Any, List, Dict

import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

# 루트 및 src 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import (
        YOLO_MODEL_PATH,
        SIGN_MODEL_PATH,
        LABEL_MAP_PATH
    )
    from src.modules.model import SignLanguageClassifier
    from src.modules.features import normalize_keypoints, extract_features_with_velocity
except ImportError:
    from config import (
        YOLO_MODEL_PATH,
        SIGN_MODEL_PATH,
        LABEL_MAP_PATH
    )
    from modules.model import SignLanguageClassifier
    from modules.features import normalize_keypoints, extract_features_with_velocity

FIXED_SEQ_LEN = 30
INPUT_DIM = 102
SLIDING_WINDOW_SIZE = 30
SLIDING_STEP = 5
CONFIDENCE_THRESHOLD = 0.65


def draw_korean_text(img, text, position, font_size=24, color=(255, 255, 255)):
    """PIL을 활용한 OpenCV 이미지 상 한글 출력 유틸리티"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("malgun.ttf", font_size) # Windows 기본 폰트
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class AsyncRefiner:
    def __init__(self):
        self.word_queue: queue.Queue = queue.Queue()
        self.current_sentence: str = "수어 동작을 대기 중입니다..."
        self.is_running: bool = True
        
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def add_word(self, word: str):
        self.word_queue.put(word)

    def _process_queue(self):
        recent_words: List[str] = []
        last_input_time = time.time()

        while self.is_running:
            try:
                word = self.word_queue.get(timeout=0.4)
                if not recent_words or recent_words[-1] != word:
                    recent_words.append(word)
                last_input_time = time.time()
            except queue.Empty:
                if recent_words and (time.time() - last_input_time > 1.2):
                    self.current_sentence = " ".join(recent_words) + "."
                    recent_words.clear()


def run_realtime_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 추론 디바이스: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    if not LABEL_MAP_PATH.exists() or not SIGN_MODEL_PATH.exists():
        print("[!] 학습된 모델 파일 또는 label_map.json이 존재하지 않습니다.")
        return

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)
        id2label: Dict[int, str] = {int(k): str(v) for k, v in id2label_raw.items()}

    num_classes = len(id2label)
    classifier = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load(SIGN_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    refiner = AsyncRefiner()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] 웹캠을 열 수 없습니다.")
        return

    frame_buffer: deque = deque(maxlen=SLIDING_WINDOW_SIZE)
    detected_words_history: deque = deque(maxlen=5)
    frame_count = 0

    print("[*] 실시간 수어 인식 시스템이 시작되었습니다. (종료: 'q' 키)")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        results: Any = yolo_model(frame, verbose=False)

        kpt_vector = np.zeros((51,), dtype=np.float32)

        if results is not None and len(results) > 0:
            first_res = results[0]
            if hasattr(first_res, "keypoints") and first_res.keypoints is not None:
                kpts_obj = first_res.keypoints
                if hasattr(kpts_obj, "data") and kpts_obj.data is not None and len(kpts_obj.data) > 0:
                    kpts_tensor = kpts_obj.data[0]
                    kpts_data = kpts_tensor.cpu().numpy()
                    kpt_vector = kpts_data.flatten()
                    
                    for pt in kpts_data:
                        x_val = float(pt[0])
                        y_val = float(pt[1])
                        conf = float(pt[2])
                        if conf > 0.5:
                            cv2.circle(frame, (int(x_val), int(y_val)), 4, (0, 255, 0), -1)

        frame_buffer.append(kpt_vector)
        frame_count += 1

        if len(frame_buffer) == SLIDING_WINDOW_SIZE and frame_count % SLIDING_STEP == 0:
            raw_kpts = np.array(frame_buffer, dtype=np.float32)
            norm_kpts = normalize_keypoints(raw_kpts)
            feat_102d = extract_features_with_velocity(norm_kpts)

            input_tensor = torch.tensor(feat_102d, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                max_prob, pred_idx = torch.max(probs, 1)

                prob_val = float(max_prob.item())
                idx_val = int(pred_idx.item())

                if prob_val >= CONFIDENCE_THRESHOLD:
                    pred_word = id2label[idx_val]
                    if not detected_words_history or detected_words_history[-1] != pred_word:
                        detected_words_history.append(pred_word)
                        refiner.add_word(pred_word)

        # UI 오버레이 (한글 렌더링 지원)
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        words_str = " ".join(list(detected_words_history))
        frame = draw_korean_text(frame, f"단어: {words_str}", (20, h - 80), font_size=20, color=(0, 255, 255))
        frame = draw_korean_text(frame, f"문장: {refiner.current_sentence}", (20, h - 40), font_size=24, color=(74, 222, 128))

        cv2.imshow("Real-Time Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    refiner.is_running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_system()