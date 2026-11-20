import sys
import asyncio
import base64
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

app = FastAPI()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>iPad 수어 번역 시연</title>
    <style>
        body { margin: 0; background-color: #0f172a; color: white; font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; overflow: hidden; }
        .container { position: relative; width: 92vw; max-width: 960px; background: #1e293b; border-radius: 20px; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }
        .img-wrapper { position: relative; width: 100%; padding-top: 56.25%; background: #000; }
        #stream { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; display: block; }
        .overlay { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(10px); padding: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1); }
        .words-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .words { font-size: 20px; color: #38bdf8; margin-bottom: 10px; font-weight: 600; min-height: 28px; }
        .sentence-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .sentence { font-size: 26px; color: #4ade80; font-weight: bold; min-height: 36px; }
        .status-badge { position: absolute; top: 15px; left: 15px; z-index: 10; background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(5px); padding: 6px 12px; border-radius: 20px; font-size: 13px; display: flex; align-items: center; gap: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
        .dot.online { background: #22c55e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="status-badge">
            <div class="dot" id="dot"></div>
            <span id="status-text">연결 중...</span>
        </div>
        <div class="img-wrapper">
            <img id="stream" alt="Live Feed">
        </div>
        <div class="overlay">
            <div class="words-title">Detected Glosses</div>
            <div class="words" id="words">대기 중...</div>
            <div class="sentence-title">Refined Sentence</div>
            <div class="sentence" id="sentence">수어 동작을 인식하면 문장이 생성됩니다.</div>
        </div>
    </div>
    <script>
        const streamImg = document.getElementById("stream");
        const wordsDiv = document.getElementById("words");
        const sentenceDiv = document.getElementById("sentence");
        const dot = document.getElementById("dot");
        const statusText = document.getElementById("status-text");

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            dot.classList.add("online");
            statusText.innerText = "LIVE STREAM";
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.image) {
                    streamImg.src = "data:image/jpeg;base64," + data.image;
                }
                wordsDiv.innerText = data.words || "-";
                sentenceDiv.innerText = data.sentence || "...";
            } catch (e) {
                console.error("JSON 파싱 오류:", e);
            }
        };

        ws.onclose = () => {
            dot.classList.remove("online");
            statusText.innerText = "연결 끊김";
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
    
    if not LABEL_MAP_PATH.exists() or not SIGN_MODEL_PATH.exists():
        await websocket.close(code=1008, reason="Model file missing")
        return

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)
        id2label: Dict[int, str] = {int(k): str(v) for k, v in id2label_raw.items()}

    classifier = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(id2label)).to(device)
    classifier.load_state_dict(torch.load(SIGN_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] 백엔드 카메라(0번)를 열 수 없습니다.")
        await websocket.close(code=1011, reason="Camera opening failed")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_buffer: deque = deque(maxlen=SLIDING_WINDOW_SIZE)
    detected_words: deque = deque(maxlen=5)
    frame_count = 0
    current_sentence = "수어 동작을 대기 중입니다..."

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

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
                                cv2.circle(frame, (int(x_val), int(y_val)), 4, (56, 189, 248), -1)

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

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "image": jpg_as_text,
                "words": " ".join(list(detected_words)),
                "sentence": current_sentence
            })
            
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("[*] 아이패드 WebSocket 연결 정상 해제")
    except Exception as e:
        print(f"[!] 스트리밍 중 예외 발생: {e}")
    finally:
        cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)