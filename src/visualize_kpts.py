import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# COCO 17개 관절 연결선 (상체 및 팔 중심)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 얼굴
    (5, 6),                                # 양어깨
    (5, 7), (7, 9),                        # 왼팔 (어깨 -> 팔꿈치 -> 손목)
    (6, 8), (8, 10),                       # 오른팔 (어깨 -> 팔꿈치 -> 손목)
    (5, 11), (6, 12), (11, 12),            # 몸통
]

WRIST_INDICES = {9, 10}  # 수어에서 가장 중요한 양손목 강조용


def draw_korean_text(img: np.ndarray, text: str, pos: tuple, font_size: int = 22, color=(0, 255, 255)) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_active_gloss(gestures: List[dict], current_time: float) -> str:
    """현재 재생 시간(초)에 해당하는 수어 단어 라벨 반환"""
    active = [
        g.get("gloss_id", "")
        for g in gestures
        if float(g.get("start", 0.0)) <= current_time <= float(g.get("end", 0.0))
    ]
    return ", ".join(filter(None, active)) if active else "(대기/전환 구간)"


def visualize_keypoints(target_stem: Optional[str] = None, conf_thresh: float = 0.4):
    npy_files = sorted(PROCESSED_DATA_DIR.glob("*.npy"))
    if target_stem:
        npy_files = [p for p in npy_files if p.stem == target_stem]

    if not npy_files:
        print(f"[!] '{PROCESSED_DATA_DIR}' 경로에 시각화할 .npy 파일이 없습니다.")
        return

    # 라벨 매핑용 JSON 로드 (존재할 경우)
    merged_json_path = PROCESSED_DATA_DIR / "merged_nikl_data.json"
    merged_data: Dict[str, dict] = {}
    if merged_json_path.exists():
        with open(merged_json_path, "r", encoding="utf-8") as f:
            merged_data = json.load(f)

    print("[*] 조작법: [Space] 일시정지/재생 | [N] 다음 영상 | [Q] 종료")

    for npy_path in npy_files:
        stem = npy_path.stem
        video_matches = list(RAW_DATA_DIR.rglob(f"{stem}.mp4"))
        if not video_matches:
            print(f"[!] 원본 영상을 찾을 수 없어 건너뜁니다: {stem}.mp4")
            continue

        video_path = video_matches[0]
        kpts_seq = np.load(npy_path)  # (Frames, 51)
        gestures = (
            merged_data.get(stem, {})
            .get("content", {})
            .get("sign_script", {})
            .get("sign_gestures_strong", [])
        )

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = len(kpts_seq)
        frame_idx = 0
        paused = False

        print(f"\n[>] 재생 중: {stem}.mp4 (총 {total_frames} 프레임, {fps:.1f} FPS)")

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret or frame_idx >= total_frames:
                    break

                kpts = kpts_seq[frame_idx].reshape(17, 3)

                # 1. 관절 연결선(뼈대) 그리기
                for idx1, idx2 in SKELETON_EDGES:
                    x1, y1, c1 = kpts[idx1]
                    x2, y2, c2 = kpts[idx2]
                    if c1 > conf_thresh and c2 > conf_thresh:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)

                # 2. 관절 키포인트 점 그리기 (손목은 빨간색, 나머지 관절은 초록색)
                for idx, (x, y, conf) in enumerate(kpts):
                    if conf > conf_thresh:
                        color = (0, 0, 255) if idx in WRIST_INDICES else (0, 255, 0)
                        radius = 6 if idx in WRIST_INDICES else 4
                        cv2.circle(frame, (int(x), int(y)), radius, color, -1)

                # 3. 상태 및 정답 라벨 오버레이
                current_time = frame_idx / fps
                gloss_text = get_active_gloss(gestures, current_time)

                cv2.rectangle(frame, (10, 10), (520, 95), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"File: {stem} | Frame: {frame_idx}/{total_frames} ({current_time:.2f}s)",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                frame = draw_korean_text(frame, f"라벨 단어: {gloss_text}", (20, 55), font_size=22, color=(74, 222, 128))

                cv2.imshow("Keypoint Analysis Viewer", frame)
                frame_idx += 1

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                break
            elif key == ord(" "):
                paused = not paused

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="추출된 관절 키포인트(.npy) 시각화 도구")
    parser.add_argument("--name", type=str, default=None, help="특정 파일명(확장자 제외)만 지정하여 확인")
    parser.add_argument("--conf", type=float, default=0.4, help="표시할 관절 최소 신뢰도 (기본값: 0.4)")
    args = parser.parse_args()

    visualize_keypoints(target_stem=args.name, conf_thresh=args.conf)