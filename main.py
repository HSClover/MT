import sys
import json
from pathlib import Path
from collections import deque
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    YOLO_MODEL_PATH,
    SIGN_MODEL_PATH,
    LABEL_MAP_PATH,
    INPUT_DIM,
    SLIDING_WINDOW_SIZE,
    SLIDING_STEP,
    CONFIDENCE_THRESHOLD,
)
from src.modules import (
    SignLanguageClassifier,
    normalize_keypoints,
    extract_features_with_velocity,
    AsyncSignRefiner,
)


def draw_korean_text(img: np.ndarray, text: str, position: tuple, font_size: int = 22, color=(255, 255, 255)) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def run_realtime_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 추론 디바이스: {device}")

    if not LABEL_MAP_PATH.exists() or not SIGN_MODEL_PATH.exists():
        print("[!] 학습된 모델(sign_word_model.pt) 또는 label_map.json이 없습니다. 먼저 python src/pipeline.py를 실행하세요.")
        return

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label: Dict[int, str] = {int(k): str(v) for k, v in json.load(f).items()}

    classifier = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(id2label)).to(device)
    classifier.load_state_dict(torch.load(SIGN_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    refiner = AsyncSignRefiner(use_llm=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] 웹캠을 열 수 없습니다.")
        refiner.stop()
        return

    frame_buffer: deque = deque(maxlen=SLIDING_WINDOW_SIZE)
    detected_words_history: deque = deque(maxlen=5)
    frame_count = 0

    print("[*] 실시간 수어 인식 시작 (종료: 'q')")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            results: Any = yolo_model(frame, verbose=False)
            kpt_vector = np.zeros((51,), dtype=np.float32)

            if results and len(results) > 0 and getattr(results[0], "keypoints", None) is not None:
                kpts_obj = results[0].keypoints
                if kpts_obj.data is not None and len(kpts_obj.data) > 0:
                    kpts_data = kpts_obj.data[0].cpu().numpy()
                    kpt_vector = kpts_data.flatten()
                    for pt in kpts_data:
                        if float(pt[2]) > 0.5:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            frame_buffer.append(kpt_vector)
            frame_count += 1

            if len(frame_buffer) == SLIDING_WINDOW_SIZE and frame_count % SLIDING_STEP == 0:
                norm_kpts = normalize_keypoints(np.array(frame_buffer, dtype=np.float32))
                feat_102d = extract_features_with_velocity(norm_kpts)
                input_tensor = torch.tensor(feat_102d, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = torch.softmax(classifier(input_tensor), dim=1)
                    max_prob, pred_idx = torch.max(probs, 1)
                    if float(max_prob.item()) >= CONFIDENCE_THRESHOLD:
                        pred_word = id2label[int(pred_idx.item())]
                        if not detected_words_history or detected_words_history[-1] != pred_word:
                            detected_words_history.append(pred_word)
                            refiner.push_word(pred_word)

            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            frame = draw_korean_text(frame, f"단어: {' '.join(detected_words_history)}", (20, h - 85), 20, (0, 255, 255))
            frame = draw_korean_text(frame, f"문장: {refiner.get_sentence()}", (20, h - 45), 24, (74, 222, 128))
            cv2.imshow("Real-Time Sign Language Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        refiner.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_system()