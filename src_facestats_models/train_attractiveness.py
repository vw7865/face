"""
train_attractiveness.py
-------------------------
Trains the attractiveness regressor using CLIP embeddings.

Responsibilities:
- Load labeled training set
- Train PyTorch MLP
- Save model + learning curves
"""

import torch
import polars as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .attractiveness_model import AttractivenessRegressor

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.X = df["embedding"].to_list()
        self.y = df["attractiveness"].to_list()

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def run_training():
    df = pl.read_parquet("data/embeddings/train_embeddings.parquet")

    ds = EmbeddingDataset(df)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = AttractivenessRegressor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for epoch in range(10):
        running = 0.0
        for X, y in dl:
            opt.zero_grad()
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += loss.item()
        losses.append(running / len(dl))
        print(f"Epoch {epoch}: loss={losses[-1]:.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "attractiveness_mlp.pth")
    print("Saved attractiveness model.")

    # TODO: save learning curves to disk


if __name__ == "__main__":
    run_training()
