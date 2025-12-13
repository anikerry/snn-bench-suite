import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 4000,
    timesteps: int = 20,
    n_features: int = 3,
    n_classes: int = 3,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temporal patterns with low-amplitude signals to encourage sparse firing."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1.5, timesteps)

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        base = np.stack(
            [
                0.5 * np.sin(2 * np.pi * (1 + k) * t),
                0.5 * np.cos(2 * np.pi * (1 + k) * t),
                0.3 * np.sin(3 * np.pi * (1 + k) * t),
            ],
            axis=-1,
        )

        for _ in range(per_class):
            noise = 0.03 * rng.randn(timesteps, n_features)
            seq = base + noise
            X.append(seq)
            y.append(k)

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
