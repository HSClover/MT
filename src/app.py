import sys
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

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

app = FastAPI()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>iPad 수어 번역기</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0; background-color: #0f172a; color: white;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 100vh; overflow: hidden;
        }
        .container {
            position: relative; width: 94vw; max-width: 960px;
            background: #1e293b; border-radius: 20px; overflow: hidden;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6);
        }
        .camera-wrapper {
            position: relative; width: 100%; padding-top: 65%; background: #000;
            /* 거울 모드: 비디오와 스켈레톤 캔버스를 함께 반전시켜 좌표 일치 */
            transform: scaleX(-1);
        }
        video, canvas#overlay-canvas {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;
        }
        .status-badge {
            position: absolute; top: 16px; left: 16px; z-index: 20;
            background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(8px);
            padding: 8px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
        }
        .dot { width: 9px; height: 9px; border-radius: 50%; background: #ef4444; }
        .dot.online { background: #22c55e; }
        .overlay-panel {
            position: absolute; bottom: 0; left: 0; right: 0; z-index: 20;
            background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
            padding: 20px 24px; border-top: 1px solid rgba(255, 255, 255, 0.12);
        }
        .label-title { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
        .words { font-size: 20px; color: #38bdf8; margin-bottom: 12px; font-weight: 600; min-height: 26px; }
        .sentence { font-size: 26px; color: #4ade80; font-weight: 700; min-height: 34px; }
        #error-banner {
            display: none; position: absolute; top: 60px; left: 16px; right: 16px; z-index: 30;
            background: rgba(220, 38, 38, 0.92); color: white; padding: 14px 18px;
            border-radius: 12px; font-size: 14px; line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="status-badge">
            <div class="dot" id="dot"></div>
            <span id="status-text">카메라 및 서버 초기화 중...</span>
        </div>
        <div id="error-banner"></div>

        <div class="camera-wrapper">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="overlay-canvas"></canvas>
        </div>

        <div class="overlay-panel">
            <div class="label-title">인식된 수어 단어 (Glosses)</div>
            <div class="words" id="words">대기 중...</div>
            <div class="label-title">번역된 문장 (Refined Sentence)</div>
            <div class="sentence" id="sentence">수어 동작을 시작하면 문장이 생성됩니다.</div>
        </div>
    </div>

    <script>
        const video = document.getElementById("video");
        const overlayCanvas = document.getElementById("overlay-canvas");
        const overlayCtx = overlayCanvas.getContext("2d");
        const wordsDiv = document.getElementById("words");
        const sentenceDiv = document.getElementById("sentence");
        const dot = document.getElementById("dot");
        const statusText = document.getElementById("status-text");
        const errorBanner = document.getElementById("error-banner");

        // 서버 전송용 오프스크린 캔버스 (속도 최적화를 위해 480x360 해상도로 축소 전송)
        const sendCanvas = document.createElement("canvas");
        sendCanvas.width = 480;
        sendCanvas.height = 360;
        const sendCtx = sendCanvas.getContext("2d");

        const SKELETON_EDGES = [
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12]
        ];

        let ws = null;
        let isWaitingResponse = false;

        function showError(msg) {
            errorBanner.style.display = "block";
            errorBanner.innerHTML = msg;
        }

        function drawSkeleton(keypoints) {
            overlayCanvas.width = video.videoWidth || 640;
            overlayCanvas.height = video.videoHeight || 480;
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            if (!keypoints || keypoints.length < 17) return;

            const w = overlayCanvas.width;
            const h = overlayCanvas.height;

            // 관절 연결선
            overlayCtx.strokeStyle = "#38bdf8";
            overlayCtx.lineWidth = 3;
            for (const [i, j] of SKELETON_EDGES) {
                const [x1, y1, c1] = keypoints[i];
                const [x2, y2, c2] = keypoints[j];
                if (c1 > 0.45 && c2 > 0.45) {
                    overlayCtx.beginPath();
                    overlayCtx.moveTo(x1 * w, y1 * h);
                    overlayCtx.lineTo(x2 * w, y2 * h);
                    overlayCtx.stroke();
                }
            }

            // 관절 점 (양손목 9, 10번은 빨간색 강조)
            keypoints.forEach(([x, y, conf], idx) => {
                if (conf > 0.45) {
                    overlayCtx.fillStyle = (idx === 9 || idx === 10) ? "#f43f5e" : "#4ade80";
                    overlayCtx.beginPath();
                    overlayCtx.arc(x * w, y * h, (idx === 9 || idx === 10) ? 7 : 5, 0, Math.PI * 2);
                    overlayCtx.fill();
                }
            });
        }

        async function startCameraAndSocket() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showError("⚠️ iPad Safari 보안 정책으로 인해 <b>https://</b> 주소로 접속해야 카메라가 켜집니다.<br>Windows PC에서 <code>ngrok http 8000</code> 실행 후 생성된 HTTPS 주소로 접속해 주세요.");
                statusText.innerText = "HTTPS 연결 필요";
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();
            } catch (err) {
                showError("⚠️ 카메라 권한이 거부되었거나 사용할 수 없습니다: " + err.message);
                statusText.innerText = "카메라 오류";
                return;
            }

            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.binaryType = "arraybuffer";

            ws.onopen = () => {
                dot.classList.add("online");
                statusText.innerText = "PC 서버 연결됨 (실시간 분석 중)";
                requestAnimationFrame(sendFrameLoop);
            };

            ws.onmessage = (event) => {
                isWaitingResponse = false;
                try {
                    const data = JSON.parse(event.data);
                    if (data.keypoints) drawSkeleton(data.keypoints);
                    wordsDiv.innerText = data.words || "대기 중...";
                    sentenceDiv.innerText = data.sentence || "수어 동작을 시작하면 문장이 생성됩니다.";
                } catch (e) {
                    console.error("응답 파싱 오류:", e);
                }
            };

            ws.onclose = () => {
                dot.classList.remove("online");
                statusText.innerText = "서버 연결 끊김";
            };
        }

        function sendFrameLoop() {
            if (ws && ws.readyState === WebSocket.OPEN && !isWaitingResponse && video.readyState >= 2) {
                isWaitingResponse = true;
                sendCtx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
                sendCanvas.toBlob((blob) => {
                    if (blob && ws.readyState === WebSocket.OPEN) {
                        ws.send(blob);
                    } else {
                        isWaitingResponse = false;
                    }
                }, "image/jpeg", 0.65);
            }
            setTimeout(() => requestAnimationFrame(sendFrameLoop), 40); // 최대 ~25 FPS 제한
        }

        startCameraAndSocket();
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not LABEL_MAP_PATH.exists() or not SIGN_MODEL_PATH.exists():
        await websocket.close(code=1008, reason="Model file missing")
        return

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label: Dict[int, str] = {int(k): str(v) for k, v in json.load(f).items()}

    classifier = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(id2label)).to(device)
    classifier.load_state_dict(torch.load(SIGN_MODEL_PATH, map_location=device))
    classifier.eval()

    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    refiner = AsyncSignRefiner(use_llm=True)

    frame_buffer: deque = deque(maxlen=SLIDING_WINDOW_SIZE)
    detected_words: deque = deque(maxlen=5)
    frame_count = 0

    print("[*] iPad 클라이언트 연결됨 - 영상 수신 대기 중")

    try:
        while True:
            # iPad로부터 JPEG 바이너리 프레임 수신
            img_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            h, w = frame.shape[:2]
            results: Any = yolo_model(frame, verbose=False)

            kpt_vector = np.zeros((51,), dtype=np.float32)
            ui_keypoints: List[List[float]] = []

            if results and len(results) > 0 and getattr(results[0], "keypoints", None) is not None:
                kpts_obj = results[0].keypoints
                if kpts_obj.data is not None and len(kpts_obj.data) > 0:
                    kpts_data = kpts_obj.data[0].cpu().numpy()  # (17, 3)
                    kpt_vector = kpts_data.flatten()
                    # 아이패드 화면 해상도에 무관하게 그릴 수 있도록 0~1 비율 좌표로 변환
                    ui_keypoints = [
                        [round(float(pt[0]) / w, 4), round(float(pt[1]) / h, 4), round(float(pt[2]), 3)]
                        for pt in kpts_data
                    ]

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
                        word = id2label[int(pred_idx.item())]
                        if not detected_words or detected_words[-1] != word:
                            detected_words.append(word)
                            refiner.push_word(word)

            # 아이패드로 관절 좌표 및 번역 텍스트만 회신
            await websocket.send_json({
                "keypoints": ui_keypoints,
                "words": " ".join(detected_words),
                "sentence": refiner.get_sentence(),
            })

    except WebSocketDisconnect:
        print("[*] iPad 클라이언트 연결 종료")
    except Exception as e:
        print(f"[!] 처리 중 예외 발생: {e}")
    finally:
        refiner.stop()