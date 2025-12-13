import torch

def spike_rate(spk: torch.Tensor) -> float:
    """Compute average spike rate per neuron per sample.

    spk is expected to be (T, B, C) or (B, C).
    """
    if spk.dim() == 2:
        spikes = spk
    elif spk.dim() == 3:
        spikes = spk.sum(dim=0)
    else:
        raise ValueError("Spike tensor must be 2D or 3D.")

    return float(spikes.mean().item())

