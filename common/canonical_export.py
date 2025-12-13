import json
from typing import Any, Dict
import torch
import torch.nn as nn

def _linear_layer_to_dict(layer: nn.Linear, name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "linear",
        "in_features": int(layer.in_features),
        "out_features": int(layer.out_features),
        "weights": layer.weight.detach().cpu().tolist(),
        "bias": None if layer.bias is None else layer.bias.detach().cpu().tolist(),
    }

def export_canonical(model: nn.Module, cfg: Any, path: str) -> None:
    """Export a model into a simple, hardware-agnostic canonical JSON format.

    The export focuses on fully connected layers; additional layer types
    can be added as needed.
    """
    layers: list[Dict[str, Any]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append(_linear_layer_to_dict(module, name))

    blob: Dict[str, Any] = {
        "benchmark_id": getattr(cfg, "benchmark_id", "unknown"),
        "series": getattr(cfg, "series", "X"),
        "architecture": model.__class__.__name__,
        "timesteps": getattr(cfg, "timesteps", None),
        "precision": {
            "weight_bits": getattr(cfg, "weight_bits", None),
            "signed": getattr(cfg, "signed", None),
        },
        "layers": layers,
        "metadata": {
            "task": getattr(cfg, "task", None),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)

    print(f"[export] Canonical model written to {path}")
