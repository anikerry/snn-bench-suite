import torch
import torch.nn as nn
import snntorch as snn

class E1Net(nn.Module):
    """Sparse SNN with masked weights to emulate low activity."""
    def __init__(self, input_dim: int = 8, hidden: int = 32, num_classes: int = 4, timesteps: int = 20, beta: float = 0.9, sparsity: float = 0.7):
        super().__init__()
        self.timesteps = timesteps
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden, num_classes, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

        # Register binary masks enforcing sparsity
        mask1 = (torch.rand_like(self.fc1.weight) > sparsity).float()
        mask2 = (torch.rand_like(self.fc2.weight) > sparsity).float()
        self.register_buffer("mask1", mask1)
        self.register_buffer("mask2", mask2)

    def _apply_masks(self):
        with torch.no_grad():
            self.fc1.weight.mul_(self.mask1)
            self.fc2.weight.mul_(self.mask2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.timesteps, 1, 1)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError("Input to E1Net must be (B, F) or (T, B, F).")

        self._apply_masks()

        mem1 = mem2 = None
        spikes = []
        for t in range(x.size(0)):
            cur = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.fc2(spk1)
            spk2, mem2 = self.lif2(out, mem2)
            spikes.append(spk2)
        return torch.stack(spikes, dim=0)
