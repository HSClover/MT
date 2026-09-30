import asyncio
import base64
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
LABEL_MAP_PATH = MODEL_SAVE_DIR / "label_map.json"
CLASSIFIER_MODEL_PATH = MODEL_SAVE_DIR / "sign_word_model.pt"

FIXED_SEQ_LEN = 30
INPUT_DIM = 102
SLIDING_WINDOW_SIZE = 30
SLIDING_STEP = 5
CONFIDENCE_THRESHOLD = 0.65

app = FastAPI()


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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>iPad 수어 번역 시연</title>
    <style>
        body { margin: 0; background-color: #0f172a; color: white; font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; }
        .container { position: relative; width: 90vw; max-width: 960px; background: #1e293b; border-radius: 20px; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }
        img { width: 100%; display: block; }
        .overlay { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(10px); padding: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1); }
        .words-title { font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .words { font-size: 20px; color: #38bdf8; margin-bottom: 10px; font-weight: 600; }
        .sentence-title { font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .sentence { font-size: 28px; color: #4ade80; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <img id="stream" src="" alt="Live Feed">
        <div class="overlay">
            <div class="words-title">Detected Glosses</div>
            <div class="words" id="words">대기 중...</div>
            <div class="sentence-title">Refined Sentence</div>
            <div class="sentence" id="sentence">수어 동작을 인식하면 문장이 생성됩니다.</div>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById("stream").src = "data:image/jpeg;base64," + data.image;
            document.getElementById("words").innerText = data.words || "-";
            document.getElementById("sentence").innerText = data.sentence || "...";
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(HTML_TEMPLATE)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)
        id2label: Dict[int, str] = {int(k): str(v) for k, v in id2label_raw.items()}

    classifier = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(id2label)).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO("yolov8x-pose.pt")
    cap = cv2.VideoCapture(0)

    frame_buffer: deque = deque(maxlen=SLIDING_WINDOW_SIZE)
    detected_words: deque = deque(maxlen=5)
    frame_count = 0
    current_sentence = "수어 동작을 대기 중입니다..."

    try:
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
                                cv2.circle(frame, (int(x_val), int(y_val)), 5, (56, 189, 248), -1)

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
                        word = id2label[idx_val]
                        if not detected_words or detected_words[-1] != word:
                            detected_words.append(word)
                            current_sentence = " ".join(list(detected_words)) + "."

            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "image": jpg_as_text,
                "words": " ".join(list(detected_words)),
                "sentence": current_sentence
            })
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("[*] 아이패드 웹 연결 해제")
    finally:
        cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)