import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 6000,
    side: int = 4,
    n_classes: int = 4,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate toy 4x4 'image patch' classification data."""
    rng = np.random.RandomState(50)
    n_features = side * side

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        center = rng.randn(n_features) * 0.5 + k
        for _ in range(per_class):
            patch = center + 0.5 * rng.randn(n_features)
            X.append(patch)
            y.append(k)

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
