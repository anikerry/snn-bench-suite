import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 6000,
    n_features: int = 8,
    n_classes: int = 4,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(11)
    means = rng.randn(n_classes, n_features) * 3.0

    X = []
    y = []
    per_class = n_samples // n_classes
    for k in range(n_classes):
        X_k = means[k] + rng.randn(per_class, n_features)
        y_k = np.full((per_class,), k)
        X.append(X_k)
        y.append(y_k)

    X = torch.tensor(np.vstack(X), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y), dtype=torch.long)

    n_train = int(n_samples * train_split)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
