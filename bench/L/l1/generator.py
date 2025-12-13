import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 3000,
    timesteps: int = 15,
    n_features: int = 3,
    n_classes: int = 3,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic temporal patterns using phase-shifted sinusoids."""
    rng = np.random.RandomState(30)
    t = np.linspace(0, 1, timesteps)

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        phase = k * np.pi / 4.0
        base = np.stack(
            [
                np.sin(2 * np.pi * (1 + k) * t + phase),
                np.cos(2 * np.pi * (1 + k) * t + phase),
                np.sin(4 * np.pi * (1 + k) * t + 0.5 * phase),
            ],
            axis=-1,
        )  # (T, F)

        for _ in range(per_class):
            noise = 0.05 * rng.randn(timesteps, n_features)
            seq = base + noise
            X.append(seq)
            y.append(k)

    X = torch.tensor(np.stack(X), dtype=torch.float32)       # (N, T, F)
    y = torch.tensor(np.array(y), dtype=torch.long)          # (N,)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
