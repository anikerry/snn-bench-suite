import json
import torch

def export_to_canonical(model, config, path):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().tolist()

    blob = {
        "architecture": str(model.__class__.__name__),
        "timesteps": config.timesteps,
        "precision": {
            "bits": config.weight_bits,
            "signed": config.precision_signed
        },
        "weights": weights,
        "metadata": {
            "task": config.task,
            "series": config.series
        }
    }

    with open(path, "w") as f:
        json.dump(blob, f, indent=2)

    print(f"Saved canonical model → {path}")
