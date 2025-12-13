import torch
import torch.nn as nn
import snntorch as snn

class P4Net(nn.Module):
    """Deep chain SNN with multiple bottlenecks to stress routing/topology."""
    def __init__(self, input_dim: int = 10, h1: int = 64, h2: int = 32, h3: int = 32, h4: int = 16, num_classes: int = 6, timesteps: int = 20, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(h2, h3, bias=False)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc4 = nn.Linear(h3, h4, bias=False)
        self.lif4 = snn.Leaky(beta=beta)
        self.fc5 = nn.Linear(h4, num_classes, bias=False)
        self.lif5 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("Input to P4Net must be (B, F) or (T, B, F).")

        mem1 = mem2 = mem3 = mem4 = mem5 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)
            cur = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur, mem3)
            cur = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur, mem4)
            out = self.fc5(spk4)
            spk5, mem5 = self.lif5(out, mem5)
            spikes.append(spk5)
        return torch.stack(spikes, dim=0)
