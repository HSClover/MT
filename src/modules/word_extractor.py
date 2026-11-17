import os
import sys
import json
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

MERGED_JSON_PATH = PROCESSED_DATA_DIR / "merged_nikl_data.json"
FIXED_SEQ_LEN = 30       # 시퀀스 길이 (30 프레임)
INPUT_DIM = 102          # 17개 관절 x 3D 좌표 + 속도 벡터 = 102차원
HIDDEN_DIM = 128


# 1. 신경망 모델 정의
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attn(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


class SignLanguageClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = 100, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        
        out = self.fc(context)
        return out


# 2. 데이터셋 클래스
class SignDataset(Dataset):
    def __init__(self, features: List[np.ndarray], labels: List[int]):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 3. 전처리 유틸리티 (속도 계산)
def extract_features_with_velocity(kpts: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(kpts)
    if len(kpts) > 1:
        velocity[1:] = kpts[1:] - kpts[:-1]
    return np.concatenate([kpts, velocity], axis=-1)


# 4. 실 데이터 매핑 및 파싱 함수
def load_and_process_merged_data() -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
    print(f"[*] 병합 데이터셋 로딩 중: {MERGED_JSON_PATH}")
    if not MERGED_JSON_PATH.exists():
        raise FileNotFoundError(f"[!] 병합 파일이 존재하지 않습니다: {MERGED_JSON_PATH}")

    with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    raw_samples = []
    word_counter = Counter()

    print(f"[v] 데이터셋 파싱 및 실제 .npy 키포인트 매핑 시작...")

    for file_stem, item in tqdm(merged_data.items(), desc="[실데이터 특징 추출]"):
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

                # 키포인트(51차원) + 속도(51차원) = 102차원 특징 생성
                feat_102d = extract_features_with_velocity(sub_kpts)

                # 고정 길이 FIXED_SEQ_LEN(30 프레임)으로 패딩/자르기
                if len(feat_102d) < FIXED_SEQ_LEN:
                    pad_len = FIXED_SEQ_LEN - len(feat_102d)
                    feat_102d = np.pad(feat_102d, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    feat_102d = feat_102d[:FIXED_SEQ_LEN]

                raw_samples.append((feat_102d, gloss_id))
                word_counter[gloss_id] += 1

    if not raw_samples:
        print("[!] .npy 데이터와 매핑된 유효 수어 단어 샘플을 찾지 못했습니다. tracker.py를 먼저 실행해주세요.")
        return [], [], {}

    # 상위 100개 빈도 단어 클래스 선별
    top_words = [w for w, _ in word_counter.most_common(100)]
    label2id = {w: i for i, w in enumerate(top_words)}
    id2label = {i: w for i, w in enumerate(top_words)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    features = []
    labels = []
    for feat, gloss in raw_samples:
        if gloss in label2id:
            features.append(feat)
            labels.append(label2id[gloss])

    print(f"[v] 전처리 완료: 총 {len(features)}개 실제 학습 샘플, {len(label2id)}개 단어 클래스")
    return features, labels, label2id


# 5. 메인 학습 함수
def train_word_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 학습 디바이스: {device}")

    features, labels, label2id = load_and_process_merged_data()

    if not features:
        print("[!] 학습할 데이터가 없습니다.")
        return

    dataset = SignDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=WORD_CLASSIFIER_BATCH_SIZE, shuffle=True)

    model = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n[*] 실데이터 기반 단어 분류 모델 학습 시작...")
    model.train()
    for epoch in range(1, WORD_CLASSIFIER_EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
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
        print(f"Epoch [{epoch}/{WORD_CLASSIFIER_EPOCHS}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), SIGN_MODEL_PATH)
    print(f"\n[v] 학습 완료! 모델 가중치 저장 완료: {SIGN_MODEL_PATH}")


if __name__ == "__main__":
    train_word_classifier()