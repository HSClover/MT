import os
import sys
import glob
import random
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# 모듈 독립 실행 시 상위 경로(src 및 project root) 탐색 가능하도록 sys.path 추가
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# config 모듈 불러오기
try:
    from src.config import (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        YOLO_MODEL_PATH,
        MAX_SAMPLES,
        BATCH_SIZE,
        RANDOM_SEED
    )
except ImportError:
    # fallback (src 내부에서 직접 실행할 경우)
    from config import (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        YOLO_MODEL_PATH,
        MAX_SAMPLES,
        BATCH_SIZE,
        RANDOM_SEED
    )


class YOLOJointTracker:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[*] 추론 디바이스 설정: {self.device} (GPU: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'None'})")
        
        # models/ 폴더 내의 YOLOv8 Pose 모델 로드
        model_file_str = str(model_path)
        print(f"[*] YOLO 모델 로드 경로: {model_file_str}")
        self.model = YOLO(model_file_str)
        self.model.to(self.device)

    def extract_from_video(self, video_path):
        """
        단일 영상 경로를 받아 전체 프레임 (Frames, 51) 키포인트 배열 반환
        (17개 수어 관절 점 x 3D = 51)
        """
        results = self.model(
            source=str(video_path),
            stream=True,
            verbose=False,
            device=self.device
        )
        
        sequence_data = []
        for r in results:
            # Pylance 타입 체크 경고 방지
            keypoints = getattr(r, 'keypoints', None)
            
            if keypoints is not None and len(keypoints.data) > 0:
                # 첫 번째 탐지된 사람의 키포인트 (17, 3) -> [x, y, confidence]
                kpts = keypoints.data[0].cpu().numpy().flatten()
            else:
                # 탐지 실패 시 0.0으로 채움 (17 * 3 = 51)
                kpts = np.zeros(51, dtype=np.float32)
                
            sequence_data.append(kpts)

        return np.array(sequence_data, dtype=np.float32)


def find_dataset_pairs(raw_data_dir):
    """
    raw_data_dir 내에서 MP4 영상과 해당 JSON 쌍을 탐색
    """
    dataset_pairs = []
    all_mp4 = glob.glob(os.path.join(raw_data_dir, "**", "*.mp4"), recursive=True)
    main_videos = [f for f in all_mp4 if not (f.endswith("_L.mp4") or f.endswith("_R.mp4"))]

    for video_path in main_videos:
        json_path = os.path.splitext(video_path)[0] + ".json"
        if os.path.exists(json_path):
            dataset_pairs.append((video_path, json_path))

    return dataset_pairs


def process_and_save_dataset(raw_dir=RAW_DATA_DIR, save_dir=PROCESSED_DATA_DIR, 
                             max_samples=MAX_SAMPLES, batch_size=BATCH_SIZE, seed=RANDOM_SEED):
    raw_path = Path(raw_dir).resolve()
    save_path = Path(save_dir).resolve()

    print(f"[*] Raw 데이터 탐색 경로: {raw_path}")
    print(f"[*] Processed 데이터 저장 경로: {save_path}")

    pairs = find_dataset_pairs(str(raw_path))
    if not pairs:
        print(f"\n[!] 경고: '{raw_path}' 경로에서 처리할 영상/JSON 쌍을 찾지 못했습니다.")
        return

    os.makedirs(save_path, exist_ok=True)
    print(f"[*] 총 {len(pairs)}개의 메인 영상/JSON 쌍을 발견했습니다.")

    # 미처리 항목만 필터링
    existing_files = set(f.replace(".npy", "") for f in os.listdir(save_path) if f.endswith(".npy"))
    unprocessed_pairs = [
        pair for pair in pairs 
        if os.path.splitext(os.path.basename(pair[0]))[0] not in existing_files
    ]

    print(f"[*] 기존 완료 항목: {len(existing_files)}개 | 미처리 남은 항목: {len(unprocessed_pairs)}개 | 총 발견된 항목: {len(pairs)}개")

    if not unprocessed_pairs:
        print("\n[*] 처리할 새로운 미처리 영상이 없습니다.")
        return

    if seed is not None:
        random.seed(seed)
    random.shuffle(unprocessed_pairs)

    target_pairs = unprocessed_pairs
    if max_samples is not None:
        target_pairs = unprocessed_pairs[:max_samples]
        print(f"[*] 미처리 대상 중 무작위 {len(target_pairs)}개 영상을 선택했습니다.")

    # YOLO Tracker 초기화
    tracker = YOLOJointTracker(model_path=YOLO_MODEL_PATH)

    total_samples = len(target_pairs)
    total_batches = (total_samples + batch_size - 1) // batch_size
    print(f"[*] 총 {total_batches}개 배치 (배치당 {batch_size}개 영상)로 나누어 실행합니다.\n")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch = target_pairs[batch_start:batch_end]

        print(f"=== [Batch {batch_idx + 1}/{total_batches}] 키포인트 추출 진행 중 ===")

        for video_path, _ in current_batch:
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            save_npy_path = os.path.join(save_path, f"{file_name}.npy")

            print(f"  - [추출 중]: {file_name}.mp4")
            keypoints = tracker.extract_from_video(video_path)
            np.save(save_npy_path, keypoints)

    print("\n[*] 키포인트 추출 작업 완료!")


if __name__ == "__main__":
    process_and_save_dataset()