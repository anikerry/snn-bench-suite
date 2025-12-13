import torch
import torch.nn as nn
import snntorch as snn

class N2Net(nn.Module):
    """Medium neuron-count benchmark."""
    def __init__(self, input_dim: int = 8, hidden: int = 64, num_classes: int = 6, timesteps: int = 20, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden, num_classes, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("Input to N2Net must be (B, F) or (T, B, F).")

        mem1 = mem2 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.fc2(spk1)
            spk2, mem2 = self.lif2(out, mem2)
            spikes.append(spk2)

        return torch.stack(spikes, dim=0)
