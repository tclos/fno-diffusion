#!/usr/bin/env python3
"""
Compara duas funções de loss, mantendo a normalização FIXA como
normalização por amostra (Y / |Y|.max + eps) em ambos os casos:

  A) Relative L2 padrão - sem ponderação
  B) Peak-Weighted Relative L2 - pesos proporcionais a |target|^alpha

Ambos os modelos são treinados com os mesmos hiperparâmetros e split.
Ao final, gera um plot comparativo das curvas de loss e salva métricas.

Uso:
    python scripts/train_compare_loss.py
    python scripts/train_compare_loss.py --epochs 200 --alpha 2.0
"""
import os, json, argparse, h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fno_diffusion.model import make_fno_2d

DATA_PATH = "data/snl/snl_dataset.h5"
OUT_DIR   = "results_compare_loss"
EPS       = 1e-8


# ---------------------------------------------------------------------------
# Funções de loss
# ---------------------------------------------------------------------------

def relative_l2_loss(pred, target, eps=EPS):
    """Relative L2 padrão — sem ponderação."""
    diff = (pred - target).view(pred.size(0), -1)
    tgt  = target.view(target.size(0), -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()


def peak_weighted_relative_l2(pred, target, alpha=2.0, eps=EPS):
    """
    Relative L2 ponderada pelos picos.
    W = (|target| / |target|.max)^alpha
    """
    B       = pred.size(0)
    abs_tgt = target.abs()
    max_tgt = abs_tgt.amax(dim=(1,2,3), keepdim=True) + eps
    W       = (abs_tgt / max_tgt) ** alpha
    diff    = (W * (pred - target)).view(B, -1)
    tgt_w   = (W * target).view(B, -1)
    return (diff.norm(dim=1) / (tgt_w.norm(dim=1) + eps)).mean()


# ---------------------------------------------------------------------------
# Normalização por amostra (fixa nos dois experimentos)
# ---------------------------------------------------------------------------

def normalize_per_sample(Y):
    scale = Y.abs().amax(dim=(1,2,3), keepdim=True) + EPS
    return Y / scale, scale


# ---------------------------------------------------------------------------
# Loop de treinamento genérico
# ---------------------------------------------------------------------------

def train(X, Y_norm, train_idx, val_idx, loss_fn, loss_name,
          epochs, batch_size, lr, device):
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

    # Para comparison justa, também registrar a relative L2 padrão na validação
    # (independente da loss de treino usada)
    val_losses_std = []

    print(f"\n{'='*55}\n  [{loss_name}]\n{'='*55}")
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_loader.dataset)

        model.eval()
        vl = 0.0; vl_std = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vl     += loss_fn(pred, yb).item()          * xb.size(0)
                vl_std += relative_l2_loss(pred, yb).item() * xb.size(0)
        vl     /= len(val_loader.dataset)
        vl_std /= len(val_loader.dataset)

        train_losses.append(tl)
        val_losses.append(vl)
        val_losses_std.append(vl_std)
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{epochs} | "
                  f"Train: {tl:.4e} | Val: {vl:.4e} | "
                  f"Val Rel-L2: {vl_std:.4e}")

    print(f"\n  Best val loss ({loss_name}): {best_val:.4e}")
    return model, train_losses, val_losses, val_losses_std, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file",  default=DATA_PATH)
    pa.add_argument("--epochs",  type=int,   default=100)
    pa.add_argument("--batch",   type=int,   default=16)
    pa.add_argument("--lr",      type=float, default=1e-3)
    pa.add_argument("--alpha",   type=float, default=2.0,
                    help="Expoente do peso na peak-weighted loss")
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

    # Normalização por amostra (fixa)
    Y_norm, Y_scale = normalize_per_sample(Y)
    scale_mean = float(Y_scale[train_idx].mean())

    norm_params = {"mode": "per_sample", "eps": EPS, "scale_train_mean": scale_mean}
    with open(os.path.join(args.out_dir, "norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    # -- Experimento A: relative L2 padrão -------------------------------
    loss_A = relative_l2_loss
    _, tl_A, vl_A, vl_A_std, best_A = train(
        X, Y_norm, train_idx, val_idx,
        loss_A, "A — Relative L2 padrão",
        args.epochs, args.batch, args.lr, device
    )
    torch.save(_, os.path.join(args.out_dir, "model_A_rel_l2.pth"))

    # -- Experimento B: peak-weighted relative L2 -----------------------
    def loss_B(pred, target): return peak_weighted_relative_l2(pred, target, alpha=args.alpha)
    _, tl_B, vl_B, vl_B_std, best_B = train(
        X, Y_norm, train_idx, val_idx,
        loss_B, f"B — Peak-Weighted L2  (α={args.alpha})",
        args.epochs, args.batch, args.lr, device
    )
    torch.save(_, os.path.join(args.out_dir, "model_B_peak_weighted.pth"))

    # -- Plot comparativo ------------------------------------------------
    epochs_x = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Train loss (cada um na sua escala - losses diferentes não são comparáveis diretamente)
    axes[0].plot(epochs_x, tl_A, "b-",  label="A - Rel L2")
    axes[0].plot(epochs_x, tl_B, "r--", label=f"B - Peak-W (α={args.alpha})")
    axes[0].set_yscale("log"); axes[0].set_title("Train Loss (escala da própria loss)")
    axes[0].set_xlabel("Época"); axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.5)

    # Val loss na escala de cada loss
    axes[1].plot(epochs_x, vl_A, "b-",  label="A - Rel L2")
    axes[1].plot(epochs_x, vl_B, "r--", label=f"B - Peak-W (α={args.alpha})")
    axes[1].set_yscale("log"); axes[1].set_title("Val Loss (escala da própria loss)")
    axes[1].set_xlabel("Época"); axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.5)

    # Val loss em relative L2 padrão para ambos - comparison justa
    axes[2].plot(epochs_x, vl_A_std, "b-",  label="A - Rel L2")
    axes[2].plot(epochs_x, vl_B_std, "r--", label=f"B - Peak-W (α={args.alpha})")
    axes[2].set_yscale("log"); axes[2].set_title("Val Loss em Relative L2 padrão")
    axes[2].set_xlabel("Época"); axes[2].legend(); axes[2].grid(True, ls=":", alpha=0.5)

    fig.suptitle("Relative L2 vs Peak-Weighted L2  (normalização = por amostra)",
                 fontweight="bold")
    plt.tight_layout()
    out_plot = os.path.join(args.out_dir, "compare_loss.pdf")
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()

    # -- Resumo ---------------------------------------------------------
    summary = {
        "normalization":    "per_sample",
        "A_rel_l2":         {"best_val_loss": best_A,
                             "best_val_std":  min(vl_A_std)},
        "B_peak_weighted":  {"alpha": args.alpha,
                             "best_val_loss": best_B,
                             "best_val_std":  min(vl_B_std)},
        "winner_by_std":    "B" if min(vl_B_std) < min(vl_A_std) else "A",
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n- Resumo ----------------------------------------------")
    print(f"  A (Rel L2)      - best val std: {min(vl_A_std):.4e}")
    print(f"  B (Peak-W α={args.alpha}) - best val std: {min(vl_B_std):.4e}")
    print(f"  Vencedor (Rel L2 padrão): {'B - Peak-Weighted' if min(vl_B_std) < min(vl_A_std) else 'A - Rel L2'}")
    print(f"  Plot salvo em: {out_plot}")
    print(f"--------------------------------------------------------------")


if __name__ == "__main__":
    main()
