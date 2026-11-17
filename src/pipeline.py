import sys
import time
from pathlib import Path

# 모듈 경로 추가
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.merge_json import merge_all_json_files
    from src.modules.tracker import process_and_save_dataset
    from src.modules.word_extractor import train_word_classifier
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
except ImportError:
    from merge_json import merge_all_json_files
    from modules.tracker import process_and_save_dataset
    from modules.word_extractor import train_word_classifier
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def run_full_pipeline(run_step1: bool = True, run_step2: bool = True, run_step3: bool = True):
    """
    수어 번역 시스템 전체 전처리 및 모델 학습 파이프라인 실행
    """
    print("=" * 60)
    print("      수어 데이터 처리 및 모델 학습 통합 파이프라인 시작      ")
    print("=" * 60)
    start_time = time.time()

    # Step 1: 원본 JSON 데이터 병합
    if run_step1:
        print("\n[Stage 1/3] 원본 JSON 데이터 병합 시작...")
        merge_all_json_files()
    else:
        print("\n[Stage 1/3] 원본 JSON 데이터 병합 건너뜀.")

    # Step 2: YOLO 기반 관절 키포인트 추출 및 저장
    if run_step2:
        print("\n[Stage 2/3] 비디오 프레임 키포인트 추출 시작...")
        process_and_save_dataset(
            raw_dir=RAW_DATA_DIR,
            save_dir=PROCESSED_DATA_DIR
        )
    else:
        print("\n[Stage 2/3] 비디오 프레임 키포인트 추출 건너뜀.")

    # Step 3: 단어 분류 신경망 모델 학습
    if run_step3:
        print("\n[Stage 3/3] 수어 단어 분류 모델 학습 시작...")
        train_word_classifier()
    else:
        print("\n[Stage 3/3] 수어 단어 분류 모델 학습 건너뜀.")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[v] 전체 파이프라인 작업 완료! (소요 시간: {elapsed_time:.2f}초)")
    print("=" * 60)


if __name__ == "__main__":
    # 전체 파이프라인 순차 실행
    run_full_pipeline()