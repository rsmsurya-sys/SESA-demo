"""Optional 1D-CNN + BiLSTM heart-rate regressor (PyTorch).

Weights are not bundled. Train against a pulse-ox / Polar H10 reference and
save a state_dict, then pass the path to PPGPipeline(dl_hr_model_path=...).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def resample_to_n(x: np.ndarray, n: int = 900) -> np.ndarray:
    """Linear resample a 1-D PPG vector to `n` samples (model input length)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == n:
        return x
    t_old = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = np.interp(t_new, t_old, x).astype(np.float32)
    y = y - y.mean()
    std = y.std()
    if std > 1e-6:
        y = y / std
    return y


if torch is not None:

    class AttentionPool(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.score = nn.Linear(dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, C)
            w = torch.softmax(self.score(x), dim=1)
            return (w * x).sum(dim=1)

    class CNNLSTMHeartRate(nn.Module):
        """Matches the architecture document: Conv → pool → BiLSTM → BPM."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(16)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(32)
            self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=64,
                batch_first=True,
                bidirectional=True,
            )
            self.pool = AttentionPool(128)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 1, T)
            h = F.relu(self.bn1(self.conv1(x)))
            h = F.max_pool1d(h, 2)
            h = F.relu(self.bn2(self.conv2(h)))
            h = F.max_pool1d(h, 2)
            h = F.relu(self.conv3(h))
            h = h.transpose(1, 2)  # (B, T, 64)
            h, _ = self.lstm(h)
            h = self.pool(h)
            bpm = self.fc(h).squeeze(-1)
            return bpm

else:  # pragma: no cover

    class CNNLSTMHeartRate:  # type: ignore
        def __init__(self) -> None:
            raise ImportError("PyTorch is required for CNNLSTMHeartRate")
