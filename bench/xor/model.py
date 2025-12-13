import torch
import torch.nn as nn
import snntorch as snn

class XORNet(nn.Module):
    """Simple 2-4-2 SNN for XOR with rate coding over time.

    The network outputs two logits corresponding to classes 0 and 1.
    """
    def __init__(self, timesteps: int = 20, beta: float = 0.9):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(2, 4, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(4, 2, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: shape (B, 2) or (T, B, 2). If (B, 2), the input is repeated over time.
        Returns spike tensor of shape (T, B, 2).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("Input to XORNet must be (B, 2) or (T, B, 2).")

        mem1 = mem2 = None
        spikes = []

        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.fc2(spk1)
            spk2, mem2 = self.lif2(out, mem2)
            spikes.append(spk2)

        return torch.stack(spikes, dim=0)
