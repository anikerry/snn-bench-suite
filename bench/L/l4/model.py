import torch
import torch.nn as nn
import snntorch as snn

class L4Net(nn.Module):
    """Deep temporal SNN with two hidden layers for long sequences."""
    def __init__(self, input_dim: int = 4, h1: int = 48, h2: int = 48, num_classes: int = 5, timesteps: int = 35, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(h2, num_classes, bias=False)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input to L4Net must be (T, B, F).")

        mem1 = mem2 = mem3 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            out = self.fc3(spk2)
            spk3, mem3 = self.lif3(out, mem3)
            spikes.append(spk3)
        return torch.stack(spikes, dim=0)
