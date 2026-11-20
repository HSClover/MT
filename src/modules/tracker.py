import os
import sys
import glob
import random
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    YOLO_MODEL_PATH,
    MAX_SAMPLES,
    BATCH_SIZE,
    RANDOM_SEED,
)


class YOLOJointTracker:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] 추론 디바이스: {self.device} ({torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'})")
        self.model = YOLO(str(model_path))
        self.model.to(self.device)

    def extract_from_video(self, video_path) -> np.ndarray:
        results = self.model(source=str(video_path), stream=True, verbose=False, device=self.device)
        sequence_data = []

        for r in results:
            keypoints = getattr(r, "keypoints", None)
            if keypoints is not None and len(keypoints.data) > 0:
                kpts = keypoints.data[0].cpu().numpy().flatten()
            else:
                kpts = np.zeros(51, dtype=np.float32)
            sequence_data.append(kpts)

        return np.array(sequence_data, dtype=np.float32)


def find_dataset_pairs(raw_data_dir):
    dataset_pairs = []
    all_mp4 = glob.glob(os.path.join(raw_data_dir, "**", "*.mp4"), recursive=True)
    main_videos = [f for f in all_mp4 if not (f.endswith("_L.mp4") or f.endswith("_R.mp4"))]

    for video_path in main_videos:
        json_path = os.path.splitext(video_path)[0] + ".json"
        if os.path.exists(json_path):
            dataset_pairs.append((video_path, json_path))
    return dataset_pairs


def process_and_save_dataset(
    raw_dir=RAW_DATA_DIR,
    save_dir=PROCESSED_DATA_DIR,
    max_samples=MAX_SAMPLES,
    batch_size=BATCH_SIZE,
    seed=RANDOM_SEED,
):
    raw_path = Path(raw_dir).resolve()
    save_path = Path(save_dir).resolve()

    pairs = find_dataset_pairs(str(raw_path))
    if not pairs:
        print(f"[!] '{raw_path}' 경로에서 영상/JSON 쌍을 찾지 못했습니다.")
        return

    os.makedirs(save_path, exist_ok=True)
    existing_files = {f.replace(".npy", "") for f in os.listdir(save_path) if f.endswith(".npy")}
    unprocessed_pairs = [
        pair for pair in pairs if os.path.splitext(os.path.basename(pair[0]))[0] not in existing_files
    ]

    print(f"[*] 전체: {len(pairs)}개 | 완료: {len(existing_files)}개 | 미처리: {len(unprocessed_pairs)}개")
    if not unprocessed_pairs:
        return

    if seed is not None:
        random.seed(seed)
    random.shuffle(unprocessed_pairs)

    target_pairs = unprocessed_pairs[:max_samples] if max_samples is not None else unprocessed_pairs
    tracker = YOLOJointTracker(model_path=YOLO_MODEL_PATH)

    total_batches = (len(target_pairs) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        batch = target_pairs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(f"=== [Batch {batch_idx + 1}/{total_batches}] 추출 진행 중 ===")
        for video_path, _ in batch:
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            keypoints = tracker.extract_from_video(video_path)
            np.save(save_path / f"{file_name}.npy", keypoints)

    print("[v] 키포인트 추출 완료!")


if __name__ == "__main__":
    process_and_save_dataset()