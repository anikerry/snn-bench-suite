import torch
import torch.nn as nn
import snntorch as snn

class P2Net(nn.Module):
    """Deeper chain SNN (4 hidden layers)."""
    def __init__(self, input_dim: int = 6, h: int = 32, num_classes: int = 5, timesteps: int = 20, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, h, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(h, h, bias=False)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(h, h, bias=False)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(h, num_classes, bias=False)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("Input to P2Net must be (B, F) or (T, B, F).")

        mem1 = mem2 = mem3 = mem4 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            cur = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur, mem3)
            out = self.fc4(spk3)
            spk4, mem4 = self.lif4(out, mem4)
            spikes.append(spk4)
        return torch.stack(spikes, dim=0)
