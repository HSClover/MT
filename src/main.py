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
import torch.nn as nn
from ultralytics import YOLO

# 1. 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
LABEL_MAP_PATH = MODEL_SAVE_DIR / "label_map.json"
CLASSIFIER_MODEL_PATH = MODEL_SAVE_DIR / "sign_word_model.pt"

# 2. 전역 변수
FIXED_SEQ_LEN = 30
INPUT_DIM = 102
SLIDING_WINDOW_SIZE = 30
SLIDING_STEP = 5
CONFIDENCE_THRESHOLD = 0.65


# 3. 모델 클래스 정의
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attn(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


class SignLanguageClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        
        out = self.fc(context)
        return out


# 4. 전처리 및 특징 추출 함수
def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    normalized = kpts.copy()
    num_frames = len(normalized)
    
    for f in range(num_frames):
        frame_data = normalized[f].reshape(17, 3)
        left_shoulder = frame_data[5, :2]
        right_shoulder = frame_data[6, :2]
        
        if frame_data[5, 2] > 0.1 and frame_data[6, 2] > 0.1:
            center = (left_shoulder + right_shoulder) / 2.0
            scale = float(np.linalg.norm(left_shoulder - right_shoulder))
            if scale < 1e-5:
                scale = 1.0
        else:
            center = np.mean(frame_data[:, :2], axis=0)
            scale = 1.0

        frame_data[:, :2] = (frame_data[:, :2] - center) / scale
        normalized[f] = frame_data.flatten()
        
    return normalized


def extract_features_with_velocity(kpts: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(kpts)
    velocity[1:] = kpts[1:] - kpts[:-1]
    combined = np.concatenate([kpts, velocity], axis=-1)
    return combined


# 5. 비동기 문맥 정제 스레드
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


# 6. 메인 실시간 추론 루프
def run_realtime_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 추론 디바이스: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    if not LABEL_MAP_PATH.exists() or not CLASSIFIER_MODEL_PATH.exists():
        print("[!] 학습된 모델 파일 또는 label_map.json이 존재하지 않습니다.")
        return

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)
        id2label: Dict[int, str] = {int(k): str(v) for k, v in id2label_raw.items()}

    num_classes = len(id2label)
    classifier = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO("yolov8x-pose.pt")
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
                    kpts_data = kpts_tensor.cpu().numpy()  # (17, 3)
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

        # UI 오버레이
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        words_str = " ".join(list(detected_words_history))
        cv2.putText(frame, f"Words: {words_str}", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {refiner.current_sentence}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Real-Time Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    refiner.is_running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_system()