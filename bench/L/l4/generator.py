import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 4800,
    timesteps: int = 35,
    n_features: int = 4,
    n_classes: int = 5,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(62)
    t = np.linspace(0, 4, timesteps)

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        freq = 0.4 + 0.3 * k
        base = np.stack(
            [
                np.sin(2 * np.pi * freq * t),
                np.cos(2 * np.pi * freq * t),
                np.sin(3 * np.pi * freq * t + 0.1 * k),
                np.cos(0.5 * np.pi * freq * t + 0.2 * k),
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
