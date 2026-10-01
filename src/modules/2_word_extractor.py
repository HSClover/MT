import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 1. 경로 및 하이퍼파라미터 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_SAVE_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

MERGED_JSON_PATH = PROCESSED_DIR / "merged_nikl_data.json"
CLASSIFIER_MODEL_PATH = MODEL_SAVE_DIR / "sign_word_model.pt"
LABEL_MAP_PATH = MODEL_SAVE_DIR / "label_map.json"

FIXED_SEQ_LEN = 30       # 시퀀스 길이 (30 프레임)
INPUT_DIM = 102          # 17개 관절 x 3D 좌표 + 속도 벡터 = 102차원
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


# 2. 모델 구조 정의 (1D-CNN + Bi-LSTM + Temporal Attention)
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


# 3. 데이터셋 클래스 정의
class SignDataset(Dataset):
    def __init__(self, features: List[np.ndarray], labels: List[int]):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 4. 데이터 로드 및 전처리 함수
def load_and_process_merged_data() -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
    print(f"[*] 병합 데이터셋 로딩 중: {MERGED_JSON_PATH}")
    if not MERGED_JSON_PATH.exists():
        raise FileNotFoundError(f"[!] 병합 파일이 존재하지 않습니다: {MERGED_JSON_PATH}")

    with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    print(f"[v] 데이터셋 로드 완료! 총 {len(merged_data)}개 샘플 파싱 시작...")

    raw_samples = []
    word_counter = Counter()

    for file_id, item in tqdm(merged_data.items(), desc="[라벨 및 데이터 추출]"):
        content = item.get("content", {})
        sign_script = content.get("sign_script", {})
        gestures = sign_script.get("sign_gestures_strong", [])

        if not gestures:
            continue

        for g in gestures:
            gloss_id = g.get("gloss_id")
            start_time = g.get("start", 0.0)
            end_time = g.get("end", 0.0)

            if gloss_id and end_time > start_time:
                # 더미 102차원 특징 생성 (실제 데이터 구축 시 추출된 Keypoint 배열과 매핑)
                num_frames = max(1, int((end_time - start_time) * 30))
                dummy_kpt = np.random.randn(num_frames, 102).astype(np.float32)
                
                # 시퀀스 길이를 FIXED_SEQ_LEN(30)으로 맞춤 (Zero-padding 또는 Truncate)
                if len(dummy_kpt) < FIXED_SEQ_LEN:
                    pad_len = FIXED_SEQ_LEN - len(dummy_kpt)
                    dummy_kpt = np.pad(dummy_kpt, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    dummy_kpt = dummy_kpt[:FIXED_SEQ_LEN]

                raw_samples.append((dummy_kpt, gloss_id))
                word_counter[gloss_id] += 1

    # 상위 N개 클래스 선별 (예: 빈도수 상위 100개 단어)
    top_words = [w for w, _ in word_counter.most_common(100)]
    label2id = {w: i for i, w in enumerate(top_words)}
    id2label = {i: w for i, w in enumerate(top_words)}

    # label_map.json 저장
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    features = []
    labels = []
    for feat, gloss in raw_samples:
        if gloss in label2id:
            features.append(feat)
            labels.append(label2id[gloss])

    print(f"[v] 전처리 완료: 총 {len(features)}개 학습 샘플, {len(label2id)}개 단어 클래스")
    return features, labels, label2id


# 5. 메인 학습 실행
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 학습 디바이스: {device}")

    features, labels, label2id = load_and_process_merged_data()

    if not features:
        print("[!] 학습할 샘플이 없습니다.")
        return

    dataset = SignDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n[*] 모델 학습 시작...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
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
        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 모델 가중치 저장
    torch.save(model.state_dict(), CLASSIFIER_MODEL_PATH)
    print(f"\n[v] 학습 완료! 모델 가중치 저장 완료: {CLASSIFIER_MODEL_PATH}")


if __name__ == "__main__":
    main()