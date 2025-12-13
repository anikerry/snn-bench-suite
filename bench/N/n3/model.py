import torch
import torch.nn as nn
import snntorch as snn

class N3Net(nn.Module):
    """Larger neuron-count benchmark approaching typical small-hardware limits."""
    def __init__(self, input_dim: int = 16, hidden1: int = 128, hidden2: int = 128, num_classes: int = 8, timesteps: int = 20, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, hidden1, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(hidden2, num_classes, bias=False)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("Input to N3Net must be (B, F) or (T, B, F).")

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
