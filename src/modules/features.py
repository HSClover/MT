"""공통 키포인트 전처리 유틸리티.

학습(word_extractor), 실시간 추론(main, app) 모두에서 동일한 전처리를
사용하기 위한 단일 구현체입니다.
"""
import numpy as np


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """어깨 기준 정규화.

    YOLO Pose 17개 관절 인덱스 중 5(왼쪽 어깨), 6(오른쪽 어깨)를 기준으로
    중심 이동 + 스케일 정규화를 수행한다.
    """
    normalized = kpts.copy()

    for f in range(len(normalized)):
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
    """키포인트(51) + 속도 벡터(51) = 102차원 특징 생성."""
    velocity = np.zeros_like(kpts)
    if len(kpts) > 1:
        velocity[1:] = kpts[1:] - kpts[:-1]
    return np.concatenate([kpts, velocity], axis=-1)