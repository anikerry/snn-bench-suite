from pathlib import Path

from .config import get_config
from .model import N2Net
from .generator import generate_data
from common.train_manager import Trainer
from common.canonical_export import export_canonical

def main() -> None:
    cfg = get_config()
    X_train, y_train, X_test, y_test = generate_data()
    model = N2Net(timesteps=cfg.timesteps)
    trainer = Trainer(model, cfg)

    history = trainer.train(X_train, y_train)
    loss, acc = trainer.evaluate(X_test, y_test)
    print(f"[N2] Final test loss: {loss:.4f}, accuracy: {acc:.3f}")

    artifacts_dir = Path("artifacts/N2")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    export_canonical(model, cfg, str(artifacts_dir / "canonical_model.json"))

if __name__ == "__main__":
    main()
