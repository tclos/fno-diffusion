# scripts/run_training.py

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim import Adam

from fno_diffusion.data_loader import load_pdebench_heat, make_initial_final_pairs
from fno_diffusion.model import make_fno_1d
from fno_diffusion.train import train_epoch, eval_epoch


DATA_PATH = "data/1D_diff-sorp_NA_NA.h5"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    u, x, t = load_pdebench_heat(DATA_PATH)
    u0, uT = make_initial_final_pairs(u)

    dataset = TensorDataset(u0, uT)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = make_fno_1d().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.3e} | "
            f"Val: {val_loss:.3e}"
        )


if __name__ == "__main__":
    main()
