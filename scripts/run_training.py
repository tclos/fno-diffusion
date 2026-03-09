# scripts/run_training.py

import os
import json
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim import Adam

from fno_diffusion.model import make_fno_2d


DATA_PATH = "data/snl/snl_dataset.h5"
RUN_DIR = "results_snl"

CONFIG = {
    "model": {
        "type": "FNO_2D",
        "n_modes": (16, 16),
        "hidden_channels": 64,
    },
    "training": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-3,
    },
    "data": {
        "path": DATA_PATH,
    },
}

def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        X = hf["X"][:]   # (N, Nf, Nt, 1)
        Y = hf["Y"][:]   # (N, Nf, Nt, 1)

    return torch.tensor(X, dtype=torch.float32), \
           torch.tensor(Y, dtype=torch.float32) 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    os.makedirs(RUN_DIR, exist_ok=True)
    
    with open(os.path.join(RUN_DIR, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f, sort_keys=False)

    # Load dataset
    X, Y = load_snl_dataset(DATA_PATH)
    
    X = X.permute(0, 3, 1, 2)  # (N, 1, Nf, Nθ)
    Y = Y.permute(0, 3, 1, 2)  # (N, 1, Nf, Nθ)
    print("Dataset shape:", X.shape)

    dataset = TensorDataset(X, Y)

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["training"]["batch_size"],
    )
    
    # Model
    model = make_fno_2d(
        n_modes=CONFIG["model"]["n_modes"],
        hidden_channels=CONFIG["model"]["hidden_channels"],
        in_channels=1,
        out_channels=1,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"]
    )

    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(CONFIG["training"]["epochs"]):

        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.3e} | "
            f"Val: {val_loss:.3e}"
        )

    # Save model
    torch.save(model.state_dict(), os.path.join(RUN_DIR, "model.pth"))

    metrics = {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": min(val_losses),
        "epochs": CONFIG["training"]["epochs"],
        "batch_size": CONFIG["training"]["batch_size"],
        "learning_rate": CONFIG["training"]["learning_rate"],
        "device": device,
    }

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "loss_curves.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
