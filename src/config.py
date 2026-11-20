from pathlib import Path

# 1. 프로젝트 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR  # 하위 호환용 별칭

# 2. 세부 모델 및 라벨 경로
YOLO_MODEL_PATH = MODELS_DIR / "yolov8x-pose.pt"
SIGN_MODEL_PATH = MODELS_DIR / "sign_word_model.pt"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_LANDMARKER_PATH = MODELS_DIR / "pose_landmarker_full.task"

# 3. 1단계 (Tracker) 옵션
MAX_SAMPLES = 100      # 무작위 추출 영상 개수 (전체 처리 시 None)
BATCH_SIZE = 32        # 배치당 영상 개수
RANDOM_SEED = 42

# 4. 모델 및 실시간 추론 공통 하이퍼파라미터
FIXED_SEQ_LEN = 30
INPUT_DIM = 102        # 17개 관절 x 3D(51) + 속도(51)
SLIDING_WINDOW_SIZE = 30
SLIDING_STEP = 5
CONFIDENCE_THRESHOLD = 0.65

WORD_CLASSIFIER_EPOCHS = 50
WORD_CLASSIFIER_BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# 5. 3단계 자연어 교정 모델
LLM_MODEL_NAME = "gogamza/kobart-base-v2"