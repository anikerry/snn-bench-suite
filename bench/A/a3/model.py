import torch
import torch.nn as nn
import snntorch as snn

class A3Net(nn.Module):
    """Temporal application-style SNN for short sequences."""
    def __init__(self, input_dim: int = 3, hidden: int = 24, num_classes: int = 4, timesteps: int = 20, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden, num_classes, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input to A3Net must be (T, B, F).")

        mem1 = mem2 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.fc2(spk1)
            spk2, mem2 = self.lif2(out, mem2)
            spikes.append(spk2)
        return torch.stack(spikes, dim=0)
