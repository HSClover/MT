"""공통 수어 분류 모델 정의.

학습(word_extractor)과 실시간 추론(main, app) 모두에서 동일한 모델을
사용하기 위한 단일 구현체입니다.
"""
import torch
import torch.nn as nn

INPUT_DIM = 102
HIDDEN_DIM = 128


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