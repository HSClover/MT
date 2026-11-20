import sys
import time
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merge_json import merge_all_json_files
from src.modules.tracker import process_and_save_dataset
from src.modules.word_extractor import train_word_classifier


def run_full_pipeline(run_merge: bool = True, run_track: bool = True, run_train: bool = True):
    start_time = time.time()
    print("=" * 60)
    print("   수어 데이터 전처리 및 모델 학습 통합 파이프라인 시작   ")
    print("=" * 60)

    if run_merge:
        print("\n[Stage 1/3] 원본 JSON 라벨 병합")
        merge_all_json_files()

    if run_track:
        print("\n[Stage 2/3] YOLOv8 관절 키포인트(.npy) 추출")
        process_and_save_dataset()

    if run_train:
        print("\n[Stage 3/3] 수어 단어 분류 모델 학습")
        train_word_classifier()

    print(f"\n[v] 전체 파이프라인 완료 (총 소요 시간: {time.time() - start_time:.2f}초)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수어 번역 모델 통합 학습 파이프라인")
    parser.add_argument("--skip-merge", action="store_true", help="1단계 JSON 병합 건너뛰기")
    parser.add_argument("--skip-track", action="store_true", help="2단계 키포인트 추출 건너뛰기")
    parser.add_argument("--skip-train", action="store_true", help="3단계 모델 학습 건너뛰기")
    args = parser.parse_args()

    run_full_pipeline(
        run_merge=not args.skip_merge,
        run_track=not args.skip_track,
        run_train=not args.skip_train,
    )