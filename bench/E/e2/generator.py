import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 7000,
    n_features: int = 12,
    n_classes: int = 5,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(41)
    means = rng.randn(n_classes, n_features) * 1.8

    X = []
    y = []
    per_class = n_samples // n_classes
    for k in range(n_classes):
        X_k = means[k] + 0.35 * rng.randn(per_class, n_features)
        y_k = np.full((per_class,), k)
        X.append(X_k)
        y.append(y_k)

    X = torch.tensor(np.vstack(X), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y), dtype=torch.long)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
