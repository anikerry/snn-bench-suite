import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 8000,
    n_features: int = 16,
    n_classes: int = 8,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Large synthetic classification dataset.

    Slightly more classes and dimensions to stress capacity.
    """
    rng = np.random.RandomState(123)
    means = rng.randn(n_classes, n_features) * 3.5

    X = []
    y = []
    per_class = n_samples // n_classes
    for k in range(n_classes):
        X_k = means[k] + rng.randn(per_class, n_features)
        y_k = np.full((per_class,), k)
        X.append(X_k)
        y.append(y_k)

    X = np.vstack(X)
    y = np.concatenate(y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    n_train = int(n_samples * train_split)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test
