import os
import glob
import json
import random
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 1. 경로 설정
MODULES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULES_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_SAVE_DIR = PROJECT_ROOT / "models"

# 2. 전역 하이퍼파라미터 설정
FIXED_SEQ_LEN = 30 
INPUT_DIM = 102
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
FPS = 30.0
PATIENCE = 100    #성능 개선이 없을 때 조기 종료를 위한 patience 값
VAL_RATIO = 0.2
RANDOM_SEED = 42


# 3. 재현성을 위한 랜덤 시드 고정 함수
def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 4. 관절 좌표 상대 정규화 함수
def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    normalized = kpts.copy()
    num_frames = len(normalized)
    
    for f in range(num_frames):
        frame = normalized[f].reshape(17, 3)
        left_shoulder = frame[5, :2]
        right_shoulder = frame[6, :2]
        
        if frame[5, 2] > 0.1 and frame[6, 2] > 0.1:
            center = (left_shoulder + right_shoulder) / 2.0
            scale = np.linalg.norm(left_shoulder - right_shoulder)
            if scale < 1e-5:
                scale = 1.0
        else:
            center = np.mean(frame[:, :2], axis=0)
            scale = 1.0

        frame[:, :2] = (frame[:, :2] - center) / scale
        normalized[f] = frame.flatten()
        
    return normalized


# 5. 관절 변화 속도(Velocity) 결합 함수 (51d + 51d = 102d)
def extract_features_with_velocity(kpts: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(kpts)
    velocity[1:] = kpts[1:] - kpts[:-1]
    combined = np.concatenate([kpts, velocity], axis=-1)
    return combined


# 6. Keypoint Data Augmentation
def augment_keypoints(kpts: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 0.01, kpts.shape)
    return kpts + noise


# 7. Dataset 클래스 정의
class SignLanguageDataset(Dataset):
    def __init__(self, processed_dir: Path, raw_dir: Path, fixed_len: int = FIXED_SEQ_LEN):
        self.samples: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}
        
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
        self.fixed_len = fixed_len
        self.is_train = True
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        print("[*] JSON 파일 인덱싱 진행 중...")
        json_map = {}
        for jp in glob.glob(str(self.raw_dir / "**" / "*.json"), recursive=True):
            json_map[Path(jp).stem] = jp

        npy_files = glob.glob(str(self.processed_dir / "*.npy"))
        print(f"[*] 총 {len(npy_files)}개의 처리된 .npy 파일을 로드합니다.")
        
        label_set = set()
        temp_data = []

        for npy_path in tqdm(npy_files, desc="[1/2] JSON 구간별 단어 파싱 중"):
            file_stem = Path(npy_path).stem
            json_path = json_map.get(file_stem)
            if not json_path:
                continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue

            # NoneType 예외 방지: null 값이 들어있거나 리스트가 아니면 빈 리스트 처리
            sign_script = meta.get("sign_script") or {}
            gestures = sign_script.get("sign_gestures_strong") or []
            if not isinstance(gestures, list):
                gestures = []

            for gesture in gestures:
                if not isinstance(gesture, dict):
                    continue
                
                word_label = gesture.get("gloss_id")
                start_time = gesture.get("start")
                end_time = gesture.get("end")

                if word_label and start_time is not None and end_time is not None:
                    label_set.add(word_label)
                    temp_data.append((npy_path, word_label, start_time, end_time))

        sorted_labels = sorted(list(label_set))
        self.label2id = {lbl: i for i, lbl in enumerate(sorted_labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(sorted_labels)}

        print(f"\n[*] 추출된 총 수어 단어 샘플 수: {len(temp_data)}개")
        print(f"[*] 고유 수어 단어 클래스 개수: {len(self.label2id)}개")

        samples_list = []
        labels_list = []

        for npy_path, word_label, start_t, end_t in tqdm(temp_data, desc="[2/2] 구간 자르기 및 특징 추출 중"):
            try:
                kpts = np.load(npy_path)
            except Exception:
                continue
            
            if len(kpts) == 0:
                continue

            start_frame = int(start_t * FPS)
            end_frame = int(end_t * FPS)

            sliced_kpts = kpts[start_frame:end_frame]
            if len(sliced_kpts) == 0:
                continue

            sliced_kpts = normalize_keypoints(sliced_kpts)

            if len(sliced_kpts) < self.fixed_len:
                pad_len = self.fixed_len - len(sliced_kpts)
                sliced_kpts = np.pad(sliced_kpts, ((0, pad_len), (0, 0)), mode='constant')
            else:
                sliced_kpts = sliced_kpts[:self.fixed_len]

            feat_102d = extract_features_with_velocity(sliced_kpts)

            samples_list.append(feat_102d)
            labels_list.append(self.label2id[word_label])

        self.samples = np.array(samples_list, dtype=np.float32) if samples_list else np.empty((0, self.fixed_len, INPUT_DIM), dtype=np.float32)
        self.labels = np.array(labels_list, dtype=np.int64) if labels_list else np.empty((0,), dtype=np.int64)

    def __len__(self):
        return len(self.samples) if self.samples is not None else 0

    def __getitem__(self, idx):
        if self.samples is None or self.labels is None:
            raise IndexError("데이터셋 로드 안 됨")
        
        sample = self.samples[idx]
        if self.is_train and np.random.rand() > 0.5:
            sample = augment_keypoints(sample)
            
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# 8. Temporal Attention 메커니즘
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


# 9. Hybrid 1D-CNN + Bi-LSTM 분류 모델
class SignLanguageClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=100, hidden_dim=128):
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

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        
        out = self.fc(context)
        return out


# 10. 학습 파이프라인 함수
def train_model():
    set_seed(RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 학습 디바이스: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"[*] 시드 고정 완료: {RANDOM_SEED}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    full_dataset = SignLanguageDataset(PROCESSED_DIR, RAW_DIR)
    if len(full_dataset) == 0:
        print("[!] 학습할 데이터가 없습니다.")
        return

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(full_dataset.label2id)

    label_map_path = MODEL_SAVE_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(full_dataset.id2label, f, ensure_ascii=False, indent=2)

    model = SignLanguageClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
    best_model_path = MODEL_SAVE_DIR / "sign_word_model.pt"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        full_dataset.is_train = True
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        model.eval()
        full_dataset.is_train = False
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> [최고 검증 성능 갱신] '{best_model_path.name}' 저장 완료")
        else:
            patience_counter += 1
            print(f"  --> 검증 성능 미개선 ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n[!] Early Stopping 발동: {PATIENCE} Epoch 동안 Validation Loss가 개선되지 않아 학습을 종료합니다.")
            break

        scheduler.step()


if __name__ == "__main__":
    train_model()