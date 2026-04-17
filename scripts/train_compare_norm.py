#!/usr/bin/env python3
"""
Compara duas configurações de normalização, mantendo a loss FIXA
como relative L2 padrão (sem ponderação) em ambos os casos:

  A) Normalização GLOBAL  - Y* = (Y - mu_Y) / sigma_Y
     (parâmetros calculados apenas no treino)

  B) Normalização POR AMOSTRA - Y_norm[i] = Y[i] / (|Y[i]|.max + eps)

Ambos os modelos são treinados com os mesmos hiperparâmetros e split.
Ao final, gera um plot comparativo das curvas de loss e salva métricas.

Uso:
    python scripts/train_compare_norm.py
    python scripts/train_compare_norm.py --epochs 200 --batch 16 --lr 1e-3
"""
import os, json, argparse, h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fno_diffusion.model import make_fno_2d

DATA_PATH = "data/snl/snl_dataset.h5"
OUT_DIR   = "results_compare_norm"
EPS       = 1e-8


# ---------------------------------------------------------------------------
# Loss — relative L2 padrão (fixa nos dois experimentos)
# ---------------------------------------------------------------------------

def relative_l2_loss(pred, target, eps=EPS):
    diff = (pred - target).view(pred.size(0), -1)
    tgt  = target.view(target.size(0), -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------

def normalize_global(Y, train_idx):
    """Z-score com mu e sigma calculados só no treino."""
    mu    = float(Y[train_idx].mean())
    sigma = float(Y[train_idx].std()) + EPS
    return (Y - mu) / sigma, {"mode": "global", "mu_Y": mu, "sigma_Y": sigma}


def normalize_per_sample(Y):
    """Y / (|Y|.max + eps) por amostra."""
    scale = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + EPS
    return Y / scale, {"mode": "per_sample", "eps": EPS,
                        "scale_train_mean": None}   # preenchido após split


# ---------------------------------------------------------------------------
# Loop de treinamento
# ---------------------------------------------------------------------------

def train(X, Y_norm, train_idx, val_idx, epochs, batch_size, lr, device, label):
    train_ds = TensorDataset(X[train_idx], Y_norm[train_idx])
    val_ds   = TensorDataset(X[val_idx],   Y_norm[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = make_fno_2d(n_modes=(16,16), hidden_channels=64,
                        in_channels=1, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f"\n{'='*55}\n  [{label}]\n{'='*55}")
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = relative_l2_loss(model(xb), yb)
            loss.backward(); optimizer.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_loader.dataset)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += relative_l2_loss(model(xb), yb).item() * xb.size(0)
        vl /= len(val_loader.dataset)

        train_losses.append(tl); val_losses.append(vl)
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{epochs} | "
                  f"Train: {tl:.4e} | Val: {vl:.4e}")

    print(f"\n  Best val loss: {best_val:.4e}")
    return model, train_losses, val_losses, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file",  default=DATA_PATH)
    pa.add_argument("--epochs",  type=int,   default=100)
    pa.add_argument("--batch",   type=int,   default=16)
    pa.add_argument("--lr",      type=float, default=1e-3)
    pa.add_argument("--out-dir", default=OUT_DIR)
    args = pa.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Carregar
    with h5py.File(args.h5file, "r") as hf:
        X = torch.tensor(hf["X"][:], dtype=torch.float32).permute(0,3,1,2)
        Y = torch.tensor(hf["Y"][:], dtype=torch.float32).permute(0,3,1,2)

    # Split idêntico para os dois experimentos
    n_total = len(X); n_train = int(0.8 * n_total)
    idx       = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = idx[:n_train]; val_idx = idx[n_train:]

    # -- Experimento A: normalização global --------------------------------
    Y_norm_A, norm_A = normalize_global(Y, train_idx)
    _, tl_A, vl_A, best_A = train(X, Y_norm_A, train_idx, val_idx,
                                   args.epochs, args.batch, args.lr,
                                   device, "A — Normalização GLOBAL")
    torch.save(_, os.path.join(args.out_dir, "model_A_global.pth"))
    with open(os.path.join(args.out_dir, "norm_params_A.json"), "w") as f:
        json.dump(norm_A, f, indent=2)

    # -- Experimento B: normalização por amostra ---------------------------
    Y_norm_B, norm_B = normalize_per_sample(Y)
    norm_B["scale_train_mean"] = float(
        Y.abs().amax(dim=(1,2,3))[train_idx].mean()
    )
    _, tl_B, vl_B, best_B = train(X, Y_norm_B, train_idx, val_idx,
                                   args.epochs, args.batch, args.lr,
                                   device, "B — Normalização POR AMOSTRA")
    torch.save(_, os.path.join(args.out_dir, "model_B_per_sample.pth"))
    with open(os.path.join(args.out_dir, "norm_params_B.json"), "w") as f:
        json.dump(norm_B, f, indent=2)

    # -- Plot comparativo --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = range(1, args.epochs + 1)

    ax1.plot(epochs_x, tl_A, "b-",  label="A - Global")
    ax1.plot(epochs_x, tl_B, "r--", label="B - Por amostra")
    ax1.set_yscale("log"); ax1.set_title("Train Loss")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Relative L2")
    ax1.legend(); ax1.grid(True, ls=":", alpha=0.5)

    ax2.plot(epochs_x, vl_A, "b-",  label="A - Global")
    ax2.plot(epochs_x, vl_B, "r--", label="B - Por amostra")
    ax2.set_yscale("log"); ax2.set_title("Validation Loss")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Relative L2")
    ax2.legend(); ax2.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Normalização Global vs Por Amostra  (loss = Relative L2)",
                 fontweight="bold")
    plt.tight_layout()
    out_plot = os.path.join(args.out_dir, "compare_norm.pdf")
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()

    # -- Resumo ------------------------------------------------------------
    summary = {
        "A_global":     {"best_val_loss": best_A},
        "B_per_sample": {"best_val_loss": best_B},
        "winner":       "B_per_sample" if best_B < best_A else "A_global",
        "delta":        best_A - best_B,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n-- Resumo --------------------------------------")
    print(f"  A (global)     best val: {best_A:.4e}")
    print(f"  B (por amostra) best val: {best_B:.4e}")
    print(f"  Vencedor: {'B - Por amostra' if best_B < best_A else 'A - Global'}")
    print(f"  Plot salvo em: {out_plot}")
    print(f"---------------------------------------------------")


if __name__ == "__main__":
    main()
