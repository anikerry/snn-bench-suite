import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 7000,
    n_features: int = 8,
    n_classes: int = 5,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic 'multisensor' feature vectors for toy classification."""
    rng = np.random.RandomState(51)

    X = []
    y = []
    per_class = n_samples // n_classes

    for k in range(n_classes):
        center = rng.randn(n_features) * 0.6 + 0.5 * k
        for _ in range(per_class):
            vec = center + 0.6 * rng.randn(n_features)
            X.append(vec)
            y.append(k)

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
