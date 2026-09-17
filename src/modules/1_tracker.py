import os
import glob
import random
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# 전역 설정 변수
# ---------------------------------------------------------------------------
SAMPLES = 100   # 총 추출할 무작위 영상 개수 (전체 처리 시 None)
BATCH = 10      # 1개 배치당 영상 개수
RANDOM_SEED = 42

# 현재 파일 기준 프로젝트 루트(MT/) 및 모듈 디렉토리(src/modules) 절대 경로 자동 산출
MODULES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULES_DIR.parent.parent

class JointTracker:
    def __init__(
        self, 
        pose_model_path=MODULES_DIR / "pose_landmarker_full.task", 
        hand_model_path=MODULES_DIR / "hand_landmarker.task"
    ):
        # 1. Pose 모델 설정
        pose_base_options = python.BaseOptions(
            model_asset_path=str(pose_model_path),
            delegate=python.BaseOptions.Delegate.GPU
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            output_segmentation_masks=False
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # 2. Hand 모델 설정
        hand_base_options = python.BaseOptions(
            model_asset_path=str(hand_model_path),
            delegate=python.BaseOptions.Delegate.GPU
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def extract_from_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_landmarker.detect(mp_image)
        hand_result = self.hand_landmarker.detect(mp_image)

        # 1. 상체/팔 관절 (Pose indices 11~16)
        if pose_result.pose_landmarks:
            arm_indices = [11, 12, 13, 14, 15, 16]
            arm_pts = [[pose_result.pose_landmarks[0][i].x,
                        pose_result.pose_landmarks[0][i].y,
                        pose_result.pose_landmarks[0][i].z] for i in arm_indices]
        else:
            arm_pts = [[0.0, 0.0, 0.0]] * 6

        # 2. 왼손 / 오른손 관절 (각 21개)
        lh_pts = [[0.0, 0.0, 0.0]] * 21
        rh_pts = [[0.0, 0.0, 0.0]] * 21

        if hand_result.hand_landmarks and hand_result.handedness:
            for idx, handedness in enumerate(hand_result.handedness):
                label = handedness[0].category_name  # 'Left' or 'Right'
                pts = [[lm.x, lm.y, lm.z] for lm in hand_result.hand_landmarks[idx]]
                if label == 'Left':
                    lh_pts = pts
                elif label == 'Right':
                    rh_pts = pts

        return np.array(arm_pts + lh_pts + rh_pts).flatten()

    def extract_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        sequence_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_vector = self.extract_from_frame(frame)
            sequence_data.append(frame_vector)

        cap.release()
        return np.array(sequence_data)

    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()


def find_dataset_pairs(raw_data_dir):
    """
    raw_data_dir 내부를 탐색하여 정면 mp4와 json 쌍을 찾습니다.
    (폴더명 규칙에 구애받지 않도록 구성을 유연하게 확장)
    """
    dataset_pairs = []
    
    # 1. 모든 하위 폴더의 mp4 탐색
    all_mp4 = glob.glob(os.path.join(raw_data_dir, "**", "*.mp4"), recursive=True)
    # _L.mp4, _R.mp4 제외한 메인 정면 영상 선택
    main_videos = [f for f in all_mp4 if not (f.endswith("_L.mp4") or f.endswith("_R.mp4"))]

    for video_path in main_videos:
        json_path = os.path.splitext(video_path)[0] + ".json"
        if os.path.exists(json_path):
            dataset_pairs.append((video_path, json_path))

    return dataset_pairs


def process_and_save_dataset(raw_dir, save_dir, max_samples=SAMPLES, batch_size=BATCH, seed=RANDOM_SEED):
    raw_path = Path(raw_dir).resolve()
    save_path = Path(save_dir).resolve()

    print(f"[*] Raw 데이터 탐색 경로: {raw_path}")
    print(f"[*] Processed 데이터 저장 경로: {save_path}")

    pairs = find_dataset_pairs(str(raw_path))
    
    if not pairs:
        print(f"\n[!] 경고: '{raw_path}' 경로에서 처리할 mp4/json 쌍을 찾지 못했습니다.")
        print("    data/raw/ 하위에 mp4 파일과 동일한 이름의 json 파일이 존재하는지 확인해주세요.")
        return

    os.makedirs(save_path, exist_ok=True)
    print(f"[*] 총 {len(pairs)}개의 메인 영상/JSON 쌍을 발견했습니다.")

    # 미처리 항목만 필터링
    existing_files = set(f.replace(".npy", "") for f in os.listdir(save_path) if f.endswith(".npy"))
    unprocessed_pairs = [
        pair for pair in pairs 
        if os.path.splitext(os.path.basename(pair[0]))[0] not in existing_files
    ]

    print(f"[*] 이미 추출 완료된 항목: {len(existing_files)}개")
    print(f"[*] 남은 미처리 대상 항목: {len(unprocessed_pairs)}개")

    if not unprocessed_pairs:
        print("\n[*] 처리할 새로운 미처리 영상이 없습니다.")
        return

    # 미처리 항목 무작위 셔플
    if seed is not None:
        random.seed(seed)
    random.shuffle(unprocessed_pairs)

    target_pairs = unprocessed_pairs
    if max_samples is not None:
        target_pairs = unprocessed_pairs[:max_samples]
        print(f"[*] 미처리 대상 중 무작위 {len(target_pairs)}개 영상을 선택했습니다.")

    # MediaPipe 모델 객체 생성
    tracker = JointTracker()

    total_samples = len(target_pairs)
    total_batches = (total_samples + batch_size - 1) // batch_size
    print(f"[*] 총 {total_batches}개 배치 (배치당 {batch_size}개 영상)로 나누어 실행합니다.\n")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch = target_pairs[batch_start:batch_end]

        print(f"=== [Batch {batch_idx + 1}/{total_batches}] 진행 중 ({len(current_batch)}개 항목) ===")

        for video_path, _ in current_batch:
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            save_npy_path = os.path.join(save_path, f"{file_name}.npy")

            print(f"  - 추출 중: {file_name}.mp4")
            keypoints = tracker.extract_from_video(video_path)
            np.save(save_npy_path, keypoints)

    tracker.close()
    print("\n[*] 모든 키포인트 추출 및 데이터 세이브 완료!")


if __name__ == "__main__":
    # 절대 경로 산출 적용
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

    process_and_save_dataset(
        raw_dir=RAW_DATA_PATH,
        save_dir=PROCESSED_DATA_PATH,
        max_samples=SAMPLES,
        batch_size=BATCH,
        seed=RANDOM_SEED
    )