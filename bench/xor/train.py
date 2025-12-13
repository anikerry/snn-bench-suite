import os
from pathlib import Path
import torch

from .config import get_config
from .model import XORNet
from .generator import generate_xor_data
from common.train_manager import Trainer
from common.canonical_export import export_canonical

def main() -> None:
    cfg = get_config()
    X_train, y_train, X_test, y_test = generate_xor_data()

    model = XORNet(timesteps=cfg.timesteps)
    trainer = Trainer(model, cfg)

    history = trainer.train(X_train, y_train)
    loss, acc = trainer.evaluate(X_test, y_test)

    print(f"[xor] Final test loss: {loss:.4f}")
    print(f"[xor] Test accuracy (thresholded regression): {acc} (nan means not applicable)")

    # Export canonical format
    artifacts_dir = Path("artifacts/xor")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    export_canonical(model, cfg, str(artifacts_dir / "canonical_model.json"))

if __name__ == "__main__":
    main()
