import os
from pathlib import Path

# ==========================================
# 1. 프로젝트 기본 경로 설정 (Pathlib 기반)
# ==========================================
# config.py 파일의 위치 기준 상위 디렉토리를 BASE_DIR로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
WEIGHTS_DIR = BASE_DIR / "weights"

# ==========================================
# 2. 1단계 (Tracker) 데이터 전처리 옵션
# ==========================================
MAX_SAMPLES = 100    # 무작위 추출할 총 영상 개수 (전체 처리 시 None)
BATCH_SIZE = 10      # 1개 배치당 영상 개수
RANDOM_SEED = 42     # 무작위 시드 (고정 시 동일 조합, 매번 다르게 하려면 None)

# MediaPipe 모델 설정
MEDIAPIPE_MODEL_COMPLEXITY = 1  # 0: 실시간 카메라용(Fast), 1: 정밀 전처리용

# 키포인트 설정 (팔 6개 + 왼손 21개 + 오른손 21개 = 총 48개 점 x 3D = 144차원)
ARM_INDICES = [11, 12, 13, 14, 15, 16]
NUM_ARM_JOINTS = len(ARM_INDICES)  # 6
NUM_HAND_JOINTS = 21               # 손가락 21개
TOTAL_FEATURE_DIM = (NUM_ARM_JOINTS + NUM_HAND_JOINTS * 2) * 3  # 144

# ==========================================
# 3. 2단계 & 3단계 모델 하이퍼파라미터
# ==========================================
# 2단계: 단어/숙어 직역 분류 모델 (ST-GCN / Bi-LSTM / Transformer)
WORD_CLASSIFIER_EPOCHS = 50
WORD_CLASSIFIER_BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# 3단계: 자연어 교정 모델 (NLP / LLM)
LLM_MODEL_NAME = "gogamza/kobart-base-v2"