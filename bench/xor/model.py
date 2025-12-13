import torch
import torch.nn as nn
import snntorch as snn

class XORNet(nn.Module):
    def __init__(self, hidden=4, timesteps=20, beta=0.9):
        super().__init__()
        self.hidden = hidden
        self.timesteps = timesteps
        self.fc1 = nn.Linear(2, hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden, 1, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = mem2 = None
        spikes = []

        for t in range(self.timesteps):
            cur = self.fc1(x)
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.fc2(spk1)
            spk2, mem2 = self.lif2(out, mem2)
            spikes.append(spk2)

        return torch.stack(spikes)
