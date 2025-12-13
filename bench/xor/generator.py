import torch
import numpy as np
from typing import Tuple

def generate_xor_data(n_samples: int = 2000, train_split: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a simple XOR dataset.

    Returns X_train, y_train, X_test, y_test with:
    - X: (N, 2) float32
    - y: (N,) int64 with values 0 or 1
    """
    X = np.random.randint(0, 2, size=(n_samples, 2))
    y = (X[:, 0] ^ X[:, 1]).astype("int64")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    n_train = int(n_samples * train_split)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test

def generate(*args, **kwargs):
    """Generic wrapper used by the runner."""
    return generate_xor_data(*args, **kwargs)
