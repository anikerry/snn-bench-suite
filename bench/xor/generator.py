import torch
import numpy as np

def generate_xor_data(n=2000):
    X = np.random.randint(0, 2, size=(n, 2))
    y = (X[:, 0] ^ X[:, 1]).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)
