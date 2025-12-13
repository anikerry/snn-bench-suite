import torch
import torch.nn as nn
import snntorch as snn

class L1Net(nn.Module):
    """Temporal SNN for short sequences (latency focus)."""
    def __init__(self, input_dim: int = 3, hidden: int = 16, num_classes: int = 3, timesteps: int = 15, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden, num_classes, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is expected to be (T, B, F)."""
        if x.dim() != 3:
            raise ValueError("Input to L1Net must be (T, B, F).")

        mem1 = mem2 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.fc2(spk1)
            spk2, mem2 = self.lif2(out, mem2)
            spikes.append(spk2)
        return torch.stack(spikes, dim=0)
