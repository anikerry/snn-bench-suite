import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 5000,
    timesteps: int = 20,
    n_features: int = 3,
    n_classes: int = 4,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temporal patterns mimicking simple event streams."""
    rng = np.random.RandomState(52)
    t = np.linspace(0, 1, timesteps)

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        base = np.stack(
            [
                np.sin(2 * np.pi * (1 + 0.5 * k) * t),
                np.cos(2 * np.pi * (1 + 0.3 * k) * t),
                np.sin(3 * np.pi * (1 + 0.2 * k) * t),
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
