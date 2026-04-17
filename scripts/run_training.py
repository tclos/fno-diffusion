# scripts/run_training.py

import os
import json
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fno_diffusion.model import make_fno_2d


DATA_PATH = "data/snl/snl_dataset.h5"
RUN_DIR   = "results_snl"

CONFIG = {
    "model": {
        "type": "FNO_2D",
        "n_modes": (16, 16),
        "hidden_channels": 64,
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
    },
    "data": {
        "path": DATA_PATH,
    },
}

# ---------------------------------------------------------------------------
# Normalização por amostra: Y / (|Y|.max + eps)
# ---------------------------------------------------------------------------
 
EPS_NORM = 1e-8
 
def normalize_per_sample(Y, eps=EPS_NORM):
    """
    Normaliza cada amostra pelo seu próprio valor absoluto máximo:
        Y_norm[i] = Y[i] / (|Y[i]|.max + eps)
 
    Resultado em [-1, 1] para cada amostra.
    Retorna Y_norm e o tensor de escalas (N, 1, 1, 1) para desnormalizar.
    """
    # max absoluto por amostra: shape (N, 1, 1, 1)
    scale = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    return Y / scale, scale
 
 
def denormalize_per_sample(Y_norm, scale):
    """Desfaz a normalização: Y = Y_norm * scale."""
    return Y_norm * scale

# ---------------------------------------------------------------------------
# Loss ponderada pelos picos
# ---------------------------------------------------------------------------
 
def peak_weighted_relative_l2(pred, target, eps=EPS_NORM, alpha=2.0):
    """
    Relative L2 com peso proporcional a |target|^alpha em cada pixel.
 
    O peso enfatiza regiões de pico (grande |target|) e penaliza pouco
    as regiões planas próximas de zero.
 
        W[i]   = (|target[i]| / (|target[i]|.max + eps)) ^ alpha
        loss   = ||W * (pred - target)||_2 / ||W * target||_2
 
    alpha=2.0: ênfase quadrática nos picos (padrão recomendado).
    """
    B = pred.size(0)
 
    # Peso por amostra, normalizado em [0, 1]^alpha
    abs_tgt   = target.abs()
    max_tgt   = abs_tgt.amax(dim=(1, 2, 3), keepdim=True) + eps
    W         = (abs_tgt / max_tgt) ** alpha          # (B, 1, Nf, Nθ)
 
    diff      = (W * (pred - target)).view(B, -1)
    tgt_w     = (W * target).view(B, -1)
 
    return (diff.norm(dim=1) / (tgt_w.norm(dim=1) + eps)).mean()
 

def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        X = hf["X"][:]   # (N, Nf, Ntheta, 1)
        Y = hf["Y"][:]   # (N, Nf, Ntheta, 1)

    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 1, Nf, Ntheta)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 1, Nf, Ntheta)
    return X, Y


def compute_stats(tensor):
    return (
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.mean()),
        float(tensor.std()),
    )


def print_stats(name, min_, max_, mean_, std_):
    print(f"  {name}:")
    print(f"    min  = {min_:.4e}")
    print(f"    max  = {max_:.4e}")
    print(f"    mean = {mean_:.4e}")
    print(f"    std  = {std_:.4e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    os.makedirs(RUN_DIR, exist_ok=True)

    with open(os.path.join(RUN_DIR, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f, sort_keys=False)

    # Carregar dataset
    X, Y = load_snl_dataset(DATA_PATH)
    print(f"Dataset shape — X: {X.shape}  Y: {Y.shape}")

    n_total = len(X)
    n_train = int(0.8 * n_total)
 
    indices   = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    x_min, x_max, x_mean, x_std = compute_stats(X[train_idx])
    y_min, y_max, y_mean, y_std = compute_stats(Y[train_idx])

    print("\n-- Estatísticas do conjunto de treino (Y antes de normalizar) --")
    print_stats("X  [E(f,θ)]", x_min, x_max, x_mean, x_std)
    print_stats("Y  [S_nl]  ", y_min, y_max, y_mean, y_std)
    print("─---------------------------------------------------------------\n")

    # -- Normalização por amostra: Y / (|Y|.max + eps) --------------------
    # Aplicada ao dataset inteiro; a escala de cada amostra é salva no
    # TensorDataset para permitir a desnormalização na inferência.
    Y_norm, Y_scale = normalize_per_sample(Y)   # Y_scale: (N, 1, 1, 1)

    print(f"Escala |Y|.max — treino: min={Y_scale[train_idx].min():.3e}"
            f"  max={Y_scale[train_idx].max():.3e}"
            f"  mean={Y_scale[train_idx].mean():.3e}\n")

    # Incluir Y_scale no dataset para poder desnormalizar por amostra
    train_ds = TensorDataset(X[train_idx], Y_norm[train_idx], Y_scale[train_idx])
    val_ds   = TensorDataset(X[val_idx],   Y_norm[val_idx],   Y_scale[val_idx])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["training"]["batch_size"])

    # Modelo
    model = make_fno_2d(
        n_modes=CONFIG["model"]["n_modes"],
        hidden_channels=CONFIG["model"]["hidden_channels"],
        in_channels=1,
        out_channels=1,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_losses, val_losses = [], []
    best_val = float("inf")
    
    # Loop de treinamento
    for epoch in range(CONFIG["training"]["epochs"]):

        model.train()
        train_loss = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = peak_weighted_relative_l2(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += peak_weighted_relative_l2(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "model_best.pth"))

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.4e} | "
            f"Val: {val_loss:.4e}"
        )

    torch.save(model.state_dict(), os.path.join(RUN_DIR, "model.pth"))

    # -- Salvar norm_params ----------------------------------------------
    # Para inferência em amostras novas, a escala não estará disponível -
    # salvar a escala média do treino como fallback.
    scale_train_mean = float(Y_scale[train_idx].mean())
    norm_params = {
        "mode":             "per_sample",
        "eps":              EPS_NORM,
        "scale_train_mean": scale_train_mean,   # fallback para inferência
    }
    with open(os.path.join(RUN_DIR, "norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"\nnorm_params.json salvo (escala média treino = {scale_train_mean:.4e})")


    # -- Métricas ----------------------------------------------------
    metrics = {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val,
        "loss_type": "peak_weighted_relative_L2",
        "normalization": {
            "mode": "per_sample",
            "eps": EPS_NORM,
            "scale_train_mean": scale_train_mean,
        },
        "epochs": CONFIG["training"]["epochs"],
        "batch_size": CONFIG["training"]["batch_size"],
        "learning_rate": CONFIG["training"]["learning_rate"],
        "device": device,
        "train_stats": {
            "X": {"min": x_min, "max": x_max, "mean": x_mean, "std": x_std},
            "Y": {"min": y_min, "max": y_max, "mean": y_mean, "std": y_std},
        },
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
 
    # -- Curvas de loss -----------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "loss_curves.pdf"))
    plt.close()
    print(f"Curvas salvas em {RUN_DIR}/loss_curves.pdf")

if __name__ == "__main__":
    main()
