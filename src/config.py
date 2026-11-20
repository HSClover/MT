import os
from pathlib import Path

# ==========================================
# 1. 프로젝트 기본 경로 설정 (Pathlib 기반)
# ==========================================
# config.py가 src/config.py에 위치할 경우, parent.parent는 최상위 루트(MT/)를 가리킵니다.
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# [수정] 실제 폴더 이름인 'models'에 맞게 수정
MODELS_DIR = BASE_DIR / "models"

# ==========================================
# 1-1. 세부 모델 파일 경로 추가
# ==========================================
# 현재 파이프라인은 YOLOv8-Pose(17개 관절)만 사용한다.
# MediaPipe 모델 파일(hand_landmarker.task, pose_landmarker_full.task)은
# models/ 폴더에 남아있지만 코드에서 로드하지 않는다.
YOLO_MODEL_PATH = MODELS_DIR / "yolov8x-pose.pt"
SIGN_MODEL_PATH = MODELS_DIR / "sign_word_model.pt"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

# ==========================================
# 2. 1단계 (Tracker) 데이터 전처리 옵션
# ==========================================
MAX_SAMPLES = 100    # 무작위 추출할 총 영상 개수 (전체 처리 시 None)
BATCH_SIZE = 10      # 1개 배치당 영상 개수
RANDOM_SEED = 42     # 무작위 시드 (고정 시 동일 조합, 매번 다르게 하려면 None)

# ==========================================
# 2-1. YOLO Pose 키포인트 차원 설정
# ==========================================
# YOLO Pose 기본 17개 관절 x (x, y, confidence) = 51차원
YOLO_KEYPOINT_COUNT = 17
RAW_KEYPOINT_DIM = YOLO_KEYPOINT_COUNT * 3   # 51
# 키포인트(51) + 속도 벡터(51) = 102차원 (모델 입력)
FEATURE_DIM = RAW_KEYPOINT_DIM * 2            # 102
# 학습/추론 시 사용하는 고정 시퀀스 길이
SEQ_LEN = 30

# ==========================================
# 3. 2단계 & 3단계 모델 하이퍼파라미터
# ==========================================
# 2단계: 단어/숙어 직역 분류 모델 (ST-GCN / Bi-LSTM / Transformer)
WORD_CLASSIFIER_EPOCHS = 50
WORD_CLASSIFIER_BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# 3단계: 자연어 교정 모델 (NLP / LLM)
LLM_MODEL_NAME = "gogamza/kobart-base-v2"