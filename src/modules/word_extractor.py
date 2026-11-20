import os
import sys
import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 독립 실행 지원을 위한 sys.path 추가
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        SIGN_MODEL_PATH,
        LABEL_MAP_PATH,
        WORD_CLASSIFIER_EPOCHS,
        WORD_CLASSIFIER_BATCH_SIZE,
        LEARNING_RATE
    )
    from src.modules.model import SignLanguageClassifier
    from src.modules.features import normalize_keypoints, extract_features_with_velocity
except ImportError:
    from config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        SIGN_MODEL_PATH,
        LABEL_MAP_PATH,
        WORD_CLASSIFIER_EPOCHS,
        WORD_CLASSIFIER_BATCH_SIZE,
        LEARNING_RATE
    )
    from modules.model import SignLanguageClassifier
    from modules.features import normalize_keypoints, extract_features_with_velocity

MERGED_JSON_PATH = PROCESSED_DATA_DIR / "merged_nikl_data.json"
FIXED_SEQ_LEN = 30       # 시퀀스 길이 (30 프레임)
INPUT_DIM = 102          # 17개 관절 x 3D 좌표 + 속도 벡터 = 102차원
HIDDEN_DIM = 128

# 시험 실행용 상한 (0 = 제한 없음). 예: set MT_MAX_VIDEOS=3000
MAX_TRAIN_VIDEOS = int(os.environ.get("MT_MAX_VIDEOS", "0"))
VAL_SPLIT_RATIO = float(os.environ.get("MT_VAL_RATIO", "0.1"))
RANDOM_SEED = 42
WORD_CLASSIFIER_EPOCHS = int(os.environ.get("MT_EPOCHS", str(WORD_CLASSIFIER_EPOCHS)))
WORD_CLASSIFIER_BATCH_SIZE = int(os.environ.get("MT_BATCH", str(WORD_CLASSIFIER_BATCH_SIZE)))


# 1. 데이터셋 클래스
class SignDataset(Dataset):
    def __init__(self, features: List[np.ndarray], labels: List[int]):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 2. 실 데이터 매핑 및 파싱 함수
def load_and_process_merged_data() -> Tuple[List[np.ndarray], List[int], Dict[str, int], List[str]]:
    print(f"[*] 병합 데이터셋 로딩 중: {MERGED_JSON_PATH}")
    if not MERGED_JSON_PATH.exists():
        raise FileNotFoundError(f"[!] 병합 파일이 존재하지 않습니다: {MERGED_JSON_PATH}")

    with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    raw_samples = []
    word_counter = Counter()

    print(f"[v] 데이터셋 파싱 및 실제 .npy 키포인트 매핑 시작...")

    all_items = list(merged_data.items())
    if MAX_TRAIN_VIDEOS and len(all_items) > MAX_TRAIN_VIDEOS:
        print(f"[*] 시험 실행: 원본 영상 {len(all_items)}개 중 상위 {MAX_TRAIN_VIDEOS}개만 사용")
        all_items = all_items[:MAX_TRAIN_VIDEOS]

    for file_stem, item in tqdm(all_items, desc="[실데이터 특징 추출]"):
        npy_path = PROCESSED_DATA_DIR / f"{file_stem}.npy"
        
        # 실제 키포인트 추출 파일이 없는 경우 건너뀀
        if not npy_path.exists():
            continue

        try:
            full_kpts = np.load(npy_path) # (Frames, 51)
        except Exception:
            continue

        if len(full_kpts) == 0:
            continue

        content = item.get("content", {})
        sign_script = content.get("sign_script", {})
        gestures = sign_script.get("sign_gestures_strong", [])

        if not gestures:
            continue

        fps = 30.0 # 기본 비디오 FPS
        total_frames = len(full_kpts)

        for g in gestures:
            gloss_id = g.get("gloss_id")
            start_time = g.get("start", 0.0)
            end_time = g.get("end", 0.0)

            if gloss_id and end_time > start_time:
                # 시작/종료 시간에 해당하는 프레임 인덱스 계산
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)

                start_frame = max(0, min(start_frame, total_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame, total_frames))

                sub_kpts = full_kpts[start_frame:end_frame]
                
                if len(sub_kpts) == 0:
                    continue

                # 어깨 기준 정규화 후, 키포인트(51차원) + 속도(51차원) = 102차원 특징 생성
                feat_102d = extract_features_with_velocity(normalize_keypoints(sub_kpts))

                # 고정 길이 FIXED_SEQ_LEN(30 프레임)으로 패딩/자르기
                if len(feat_102d) < FIXED_SEQ_LEN:
                    pad_len = FIXED_SEQ_LEN - len(feat_102d)
                    feat_102d = np.pad(feat_102d, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    feat_102d = feat_102d[:FIXED_SEQ_LEN]

                raw_samples.append((feat_102d, gloss_id, file_stem))
                word_counter[gloss_id] += 1

    if not raw_samples:
        print("[!] .npy 데이터와 매핑된 유효 수어 단어 샘플을 찾지 못했습니다. tracker.py를 먼저 실행해주세요.")
        return [], [], {}, []

    # 상위 100개 빈도 단어 클래스 선별
    top_words = [w for w, _ in word_counter.most_common(100)]
    label2id = {w: i for i, w in enumerate(top_words)}
    id2label = {i: w for i, w in enumerate(top_words)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    features = []
    labels = []
    video_ids = []
    for feat, gloss, stem in raw_samples:
        if gloss in label2id:
            features.append(feat)
            labels.append(label2id[gloss])
            video_ids.append(stem)

    print(f"[v] 전처리 완료: 총 {len(features)}개 실제 학습 샘플, "
          f"{len(label2id)}개 단어 클래스, {len(set(video_ids))}개 원본 영상")
    return features, labels, label2id, video_ids


# 5. 평가 유틸리티
@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        total_loss += criterion(outputs, batch_y).item() * batch_x.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

        classes, counts = torch.unique(batch_y, return_counts=True)
        for c, n in zip(classes.tolist(), counts.tolist()):
            mask = batch_y == c
            class_total[c] += n
            class_correct[c] += int((preds[mask] == c).sum().item())

    model.train()
    loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    per_class = {c: 100.0 * int(class_correct[c].item()) / int(class_total[c].item())
                 for c in range(num_classes) if int(class_total[c].item()) > 0}
    return loss, acc, per_class


# 6. 메인 학습 함수
def train_word_classifier():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 학습 디바이스: {device}")

    features, labels, label2id, video_ids = load_and_process_merged_data()

    if not features:
        print("[!] 학습할 데이터가 없습니다.")
        return

    # 원본 영상 단위로 학습/검증 분리 (같은 영상의 다른 단어가 검증 세트로 새면 정확도가 부풀려짐)
    unique_videos = sorted(set(video_ids))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_videos)
    n_val_videos = max(1, int(len(unique_videos) * VAL_SPLIT_RATIO))
    val_videos = set(unique_videos[:n_val_videos])

    train_features, train_labels = [], []
    val_features, val_labels = [], []
    for feat, lab, stem in zip(features, labels, video_ids):
        if stem in val_videos:
            val_features.append(feat)
            val_labels.append(lab)
        else:
            train_features.append(feat)
            train_labels.append(lab)

    print(f"[*] 분리 완료: 학습 {len(train_features)}개 / 검증 {len(val_features)}개 "
          f"(영상 {len(unique_videos) - n_val_videos} / {n_val_videos})")

    train_loader = DataLoader(
        SignDataset(train_features, train_labels),
        batch_size=WORD_CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        SignDataset(val_features, val_labels),
        batch_size=max(WORD_CLASSIFIER_BATCH_SIZE, 64), shuffle=False, num_workers=0)

    num_classes = len(label2id)
    id2label = {i: w for w, i in label2id.items()}

    model = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[*] 실데이터 기반 단어 분류 모델 학습 시작... (클래스 {num_classes}개)")
    model.train()
    best_val_acc = -1.0

    for epoch in range(1, WORD_CLASSIFIER_EPOCHS + 1):
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100

        val_loss, val_acc, per_class = evaluate(model, val_loader, device, num_classes)

        message = (f"Epoch [{epoch}/{WORD_CLASSIFIER_EPOCHS}] - Loss: {epoch_loss:.4f} | "
                   f"Train: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f} | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SIGN_MODEL_PATH)
            message += " | 최고 가중치 저장"

        print(message + f" | {time.time() - epoch_start:.1f}s")

    _, final_val_acc, final_per_class = evaluate(model, val_loader, device, num_classes)

    print(f"\n[v] 학습 완료. 최고 검증 정확도: {best_val_acc:.2f}% | 마지막 검증 정확도: {final_val_acc:.2f}%")
    print(f"[v] 모델 가중치(검증 최고): {SIGN_MODEL_PATH}")

    weakest = sorted(final_per_class.items(), key=lambda kv: kv[1])[:10]
    print("[i] 검증 정확도 하위 10개 클래스:")
    for idx, acc in weakest:
        print(f"    {id2label.get(idx, idx)}: {acc:.1f}%")


if __name__ == "__main__":
    train_word_classifier()