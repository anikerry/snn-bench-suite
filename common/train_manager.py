from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainingConfig:
    """Configuration for training a benchmark model."""
    task: str = "classification"
    series: str = "X"
    benchmark_id: str = "UNKNOWN"
    timesteps: int = 20

    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    early_stop_patience: int = 50

    weight_bits: int = 4
    signed: bool = True

    seed: int = 123


class Trainer:
    """Minimal, reliable trainer for rate-coded SNN outputs.

    Assumption:
      model(x) -> spikes with shape (T, B, C) where C = num_classes.
    We convert to rate/logit via mean over time and apply CrossEntropyLoss once.
    This avoids multiple backward() calls on the same graph.
    """

    def __init__(self, model: torch.nn.Module, cfg: TrainingConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        torch.manual_seed(cfg.seed)

        if cfg.task != "classification":
            raise ValueError(f"Only task='classification' is supported in this benchmark suite (got {cfg.task}).")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def _to_batches(self, X: torch.Tensor, y: torch.Tensor):
        bs = self.cfg.batch_size
        n = X.shape[0]
        for i in range(0, n, bs):
            yield X[i:i+bs], y[i:i+bs]

    def _prepare_input(self, Xb: torch.Tensor) -> torch.Tensor:
        """Return input in the shape expected by models.

        - If Xb is (B, F): pass as-is (models will tile over timesteps internally).
        - If Xb is (B, T, F): convert to (T, B, F).
        """
        if Xb.dim() == 2:
            return Xb.to(self.device)
        if Xb.dim() == 3:
            return Xb.permute(1, 0, 2).contiguous().to(self.device)
        raise ValueError("X must be (N,F) or (N,T,F).")

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        self.model.train()

        X_train = X_train.detach()
        y_train = y_train.detach()

        best_loss = float("inf")
        patience = 0

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0

            # Shuffle each epoch
            idx = torch.randperm(X_train.shape[0])
            Xs = X_train[idx]
            ys = y_train[idx]

            for Xb, yb in self._to_batches(Xs, ys):
                Xb_in = self._prepare_input(Xb)
                yb = yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                spk = self.model(Xb_in)           # (T,B,C)
                if spk.dim() != 3:
                    raise RuntimeError(f"Model output must be (T,B,C), got {tuple(spk.shape)}")

                logits = spk.mean(dim=0)          # (B,C) rate-coded logits
                loss = self.criterion(logits, yb)

                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.item()) * yb.size(0)

                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.size(0))

            epoch_loss /= max(total, 1)
            acc = correct / max(total, 1)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"[{self.cfg.benchmark_id}] Epoch {epoch:4d}/{self.cfg.epochs} | loss={epoch_loss:.4f} | acc={acc:.3f}")

            if patience >= self.cfg.early_stop_patience:
                print(f"[{self.cfg.benchmark_id}] Early stopping at epoch {epoch} (best loss {best_loss:.4f}).")
                break

    @torch.no_grad()
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, float]:
        self.model.eval()
        X_test = X_test.detach()
        y_test = y_test.detach()

        total_loss = 0.0
        correct = 0
        total = 0

        for Xb, yb in self._to_batches(X_test, y_test):
            Xb_in = self._prepare_input(Xb)
            yb = yb.to(self.device)

            spk = self.model(Xb_in)
            logits = spk.mean(dim=0)
            loss = self.criterion(logits, yb)

            total_loss += float(loss.item()) * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc
