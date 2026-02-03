# scripts/run_training.py

import os
import matplotlib.pyplot as plt
import csv
import json
import yaml

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim import Adam

from fno_diffusion.data_loader import load_pdebench_heat, make_initial_final_pairs
from fno_diffusion.model import make_fno_1d
from fno_diffusion.train import train_epoch, eval_epoch


DATA_PATH = "data/1D_diff-sorp_NA_NA.h5"

RUN_DIR = "results"

CONFIG = {
    "model": {
        "type": "FNO",
        "optimizer": "Adam",
    },
    "training": {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
    },
    "data": {
        "dataset": "PDEBench",
        "equation": "1D diffusion–sorption",
        "path": "data/1D_diff-sorp_NA_NA.h5",
    },
}



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    os.makedirs(RUN_DIR, exist_ok=True)
    
    with open(os.path.join(RUN_DIR, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f, sort_keys=False)

    u, x, t = load_pdebench_heat(DATA_PATH)
    u0, uT = make_initial_final_pairs(u)

    dataset = TensorDataset(u0, uT)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["training"]["batch_size"])

    model = make_fno_1d().to(device)
    optimizer = Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    
    train_losses = []
    val_losses = []

    epochs = CONFIG["training"]["epochs"]
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.3e} | "
            f"Val: {val_loss:.3e}"
        )
    
    model_path = os.path.join(RUN_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    csv_path = os.path.join(RUN_DIR, "losses.csv")
    
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (tr, vl) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([epoch, tr, vl])
            
            
    
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
    
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/loss_curves.pdf")
    plt.close()


if __name__ == "__main__":
    main()

