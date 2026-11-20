import sys
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SIGN_MODEL_PATH,
    LABEL_MAP_PATH,
    FIXED_SEQ_LEN,
    INPUT_DIM,
    WORD_CLASSIFIER_EPOCHS,
    WORD_CLASSIFIER_BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
)

MERGED_JSON_PATH = PROCESSED_DATA_DIR / "merged_nikl_data.json"
HIDDEN_DIM = 128


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        return torch.sum(attn_weights * lstm_output, dim=1)


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
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.fc(context)


class SignDataset(Dataset):
    def __init__(self, features: List[np.ndarray], labels: List[int]):
        self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """양어깨(인덱스 5, 6) 중심 및 어깨 너비 기준으로 좌표 정규화"""
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
    """정규화된 51차원 좌표에 프레임 간 속도(51차원)를 결합해 102차원 특징 생성"""
    velocity = np.zeros_like(kpts)
    if len(kpts) > 1:
        velocity[1:] = kpts[1:] - kpts[:-1]
    return np.concatenate([kpts, velocity], axis=-1)


def load_and_process_merged_data() -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
    if not MERGED_JSON_PATH.exists():
        raise FileNotFoundError(f"[!] 병합 파일이 없습니다: {MERGED_JSON_PATH} (merge_json.py 먼저 실행)")

    with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    raw_samples: List[Tuple[np.ndarray, str]] = []
    word_counter: Counter = Counter()
    matched_npy_count = 0

    for file_stem, item in tqdm(merged_data.items(), desc="[실데이터 .npy 매핑]"):
        npy_path = PROCESSED_DATA_DIR / f"{file_stem}.npy"
        if not npy_path.exists():
            continue

        try:
            full_kpts = np.load(npy_path)  # (Frames, 51)
        except Exception:
            continue

        if full_kpts.ndim != 2 or len(full_kpts) == 0:
            continue

        matched_npy_count += 1
        content = item.get("content", {})
        fps = float(content.get("metaData", {}).get("fps", 30.0) or 30.0)
        gestures = content.get("sign_script", {}).get("sign_gestures_strong", [])
        total_frames = len(full_kpts)

        for g in gestures:
            gloss_id = g.get("gloss_id")
            start_time = float(g.get("start", 0.0))
            end_time = float(g.get("end", 0.0))

            if not gloss_id or end_time <= start_time:
                continue

            start_frame = max(0, min(int(start_time * fps), total_frames - 1))
            end_frame = max(start_frame + 1, min(int(end_time * fps), total_frames))
            sub_kpts = full_kpts[start_frame:end_frame]

            if len(sub_kpts) == 0:
                continue

            norm_kpts = normalize_keypoints(sub_kpts)
            feat_102d = extract_features_with_velocity(norm_kpts)

            if len(feat_102d) < FIXED_SEQ_LEN:
                pad_len = FIXED_SEQ_LEN - len(feat_102d)
                feat_102d = np.pad(feat_102d, ((0, pad_len), (0, 0)), mode="constant")
            else:
                feat_102d = feat_102d[:FIXED_SEQ_LEN]

            raw_samples.append((feat_102d, gloss_id))
            word_counter[gloss_id] += 1

    print(f"[*] 매칭된 .npy 파일 수: {matched_npy_count}개 | 추출된 단어 구간 수: {len(raw_samples)}개")

    if not raw_samples:
        print("[!] 학습할 유효 샘플이 없습니다. tracker.py로 .npy 키포인트를 먼저 추출해 주세요.")
        return [], [], {}

    top_words = [w for w, _ in word_counter.most_common(100)]
    label2id = {w: i for i, w in enumerate(top_words)}
    id2label = {i: w for i, w in enumerate(top_words)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    features, labels = [], []
    for feat, gloss in raw_samples:
        if gloss in label2id:
            features.append(feat)
            labels.append(label2id[gloss])

    print(f"[v] 전처리 완료: {len(features)}개 샘플, {len(label2id)}개 클래스")
    return features, labels, label2id


def train_word_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 학습 디바이스: {device}")

    features, labels, label2id = load_and_process_merged_data()
    if not features:
        return

    full_dataset = SignDataset(features, labels)
    if len(full_dataset) >= 10:
        val_size = max(1, int(len(full_dataset) * 0.2))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(RANDOM_SEED or 42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = full_dataset, None

    train_loader = DataLoader(train_dataset, batch_size=WORD_CLASSIFIER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=WORD_CLASSIFIER_BATCH_SIZE, shuffle=False) if val_dataset else None

    model = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = -1.0
    for epoch in range(1, WORD_CLASSIFIER_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_acc = (correct / total) * 100.0

        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_out = model(vx)
                    val_correct += (torch.argmax(v_out, dim=1) == vy).sum().item()
                    val_total += vy.size(0)
            val_acc = (val_correct / val_total) * 100.0
            print(f"Epoch [{epoch:02d}/{WORD_CLASSIFIER_EPOCHS}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            score_acc = val_acc
        else:
            print(f"Epoch [{epoch:02d}/{WORD_CLASSIFIER_EPOCHS}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            score_acc = train_acc

        if score_acc >= best_acc:
            best_acc = score_acc
            torch.save(model.state_dict(), SIGN_MODEL_PATH)

    print(f"[v] 학습 완료 (최고 정확도: {best_acc:.2f}%) -> {SIGN_MODEL_PATH}")


if __name__ == "__main__":
    train_word_classifier()