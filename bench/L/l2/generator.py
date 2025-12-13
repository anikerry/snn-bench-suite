import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 4000,
    timesteps: int = 25,
    n_features: int = 4,
    n_classes: int = 4,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(31)
    t = np.linspace(0, 2, timesteps)

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        freq = 1 + 0.5 * k
        phase = k * np.pi / 6.0
        base = np.stack(
            [
                np.sin(2 * np.pi * freq * t + phase),
                np.cos(2 * np.pi * freq * t),
                np.sin(4 * np.pi * freq * t + 0.3 * phase),
                np.cos(0.5 * np.pi * freq * t + phase),
            ],
            axis=-1,
        )

        for _ in range(per_class):
            noise = 0.05 * rng.randn(timesteps, n_features)
            seq = base + noise
            X.append(seq)
            y.append(k)

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
