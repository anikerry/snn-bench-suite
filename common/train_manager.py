import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.MSELoss() if config.output == "regression" else torch.nn.CrossEntropyLoss()

    def train(self, X, y):
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        hist = []

        for epoch in range(self.cfg.epochs):
            self.model.train()
            tot_loss = 0

            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                spk = self.model(xb)
                out = spk.sum(0)
                loss = self.loss_fn(out, yb)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                tot_loss += loss.item()

            hist.append(tot_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, loss={tot_loss:.4f}")

        return hist
