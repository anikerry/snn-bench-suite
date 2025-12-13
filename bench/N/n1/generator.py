import torch
import numpy as np
from typing import Tuple

def generate_data(
    n_samples: int = 4000,
    n_features: int = 4,
    n_classes: int = 4,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a simple synthetic classification dataset.

    Classes are separated by Gaussian blobs in feature space.
    """
    rng = np.random.RandomState(42)
    means = rng.randn(n_classes, n_features) * 2.0

    X = []
    y = []
    for k in range(n_classes):
        X_k = means[k] + 0.5 * rng.randn(n_samples // n_classes, n_features)
        y_k = np.full((n_samples // n_classes,), k)
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
