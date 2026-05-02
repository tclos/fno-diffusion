#!/usr/bin/env python3
"""
Treinamento fatorizado para E(f, theta) -> S_nl(f, theta).

  - O branch do campo/shape usa Relative L2 puro
  - A fatorização separa:
        a = max_{f,theta} |S_nl|
        S_tilde = S_nl / (a + eps)
  - O FNO aprende apenas o shape normalizado:
        E -> S_tilde
  - Uma FFN separada aprende a amplitude:
        E -> a
  - Na inferência:
        S_pred = a_pred * S_tilde_pred

Opcionalmente, é possível carregar um FNO já treinado para o campo normalizado
com --field-checkpoint e congelá-lo com --freeze-field, treinando apenas a rede
que prevê a amplitude a. Isso implementa literalmente a ideia de manter o
melhor modelo do campo e acrescentar só a previsão de a.
"""

import argparse
import json
import os
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
RUN_DIR = "results_snl_factorized"
EPS = 1e-8


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def relative_l2_loss(pred, target, eps=EPS):
    """Relative L2 puro para o campo normalizado S_tilde."""
    diff = (pred - target).view(pred.size(0), -1)
    tgt = target.view(target.size(0), -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()


def scale_log_mse(pred_a, true_a, eps=EPS):
    """MSE no log da amplitude, para estabilizar variações de ordem de grandeza."""
    return torch.mean((torch.log(pred_a + eps) - torch.log(true_a + eps)) ** 2)


def relative_l2_physical(pred, target, eps=EPS):
    """Relative L2 no campo físico reconstruído."""
    diff = (pred - target).view(pred.size(0), -1)
    tgt = target.view(target.size(0), -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()


# ---------------------------------------------------------------------------
# Fatorização
# ---------------------------------------------------------------------------

def factorize_target(Y, eps=EPS):
    """
    Y: (N, 1, Nf, Ntheta)
    Retorna:
      Y_tilde: Y / (a + eps)
      a:       max_abs(Y), shape (N, 1)
    """
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    Y_tilde = Y / a
    return Y_tilde, a.view(-1, 1)


# ---------------------------------------------------------------------------
# Modelo fatorizado
# ---------------------------------------------------------------------------

class ScaleHead(nn.Module):
    """FFN simples para prever a amplitude a a partir de E(f, theta)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


class FactorizedSnlModel(nn.Module):
    def __init__(self, n_modes=(16, 16), hidden_channels=64, scale_head_hidden=128):
        super().__init__()
        self.field_model = make_fno_2d(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=1,
            out_channels=1,
        )
        self.scale_model = ScaleHead(hidden=scale_head_hidden)

    def forward(self, x):
        y_tilde_pred = self.field_model(x)
        a_pred = self.scale_model(x)
        return y_tilde_pred, a_pred


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        X = hf["X"][:]
        Y = hf["Y"][:]
    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    return X, Y


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint inválido: não contém state_dict.")

    cleaned = {}
    for k, v in state.items():
        if k == "_metadata":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_field_checkpoint(field_model, checkpoint_path):
    """
    Carrega pesos de um FNO puro ou de um checkpoint fatorizado.
    Aceita chaves do tipo:
      - FNO puro: diretamente as chaves do FNO
      - fatorizado: prefixo field_model.
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = clean_state_dict(raw)

    # Caso checkpoint fatorizado: remover prefixo field_model.
    if any(k.startswith("field_model.") for k in state.keys()):
        state = {
            k[len("field_model."):]: v
            for k, v in state.items()
            if k.startswith("field_model.")
        }

    field_model.load_state_dict(state, strict=True)
    print(f"FNO do campo carregado de: {checkpoint_path}")


def compute_stats(tensor):
    return {
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_curve(train_values, val_values, out_path, ylabel, title):
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_sample_predictions(model, X, Y, Y_tilde, a, indices, out_path, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in indices:
            x = X[idx:idx+1].to(device)
            pred_tilde, pred_a = model(x)
            pred_phys = pred_tilde * pred_a.view(-1, 1, 1, 1)
            rows.append((
                int(idx),
                Y[idx, 0].cpu().numpy(),
                pred_phys[0, 0].cpu().numpy(),
                Y_tilde[idx, 0].cpu().numpy(),
                pred_tilde[0, 0].cpu().numpy(),
                float(a[idx].item()),
                float(pred_a[0, 0].cpu().item()),
            ))

    n = len(rows)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(15, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    for r, (idx, gt, pred, gt_tilde, pred_tilde, a_gt, a_pred) in enumerate(rows):
        vmax_phys = max(np.max(np.abs(gt)), np.max(np.abs(pred))) + 1e-12
        vmax_tilde = max(np.max(np.abs(gt_tilde)), np.max(np.abs(pred_tilde))) + 1e-12

        im0 = axes[r, 0].imshow(gt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax_phys, vmax=vmax_phys)
        axes[r, 0].set_title(f"idx={idx} | S_nl GT")
        im1 = axes[r, 1].imshow(pred, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax_phys, vmax=vmax_phys)
        axes[r, 1].set_title(f"S_nl pred\na_gt={a_gt:.2e}, a_pred={a_pred:.2e}")
        im2 = axes[r, 2].imshow(gt_tilde, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax_tilde, vmax=vmax_tilde)
        axes[r, 2].set_title("S_tilde GT")
        im3 = axes[r, 3].imshow(pred_tilde, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax_tilde, vmax=vmax_tilde)
        axes[r, 3].set_title("S_tilde pred")

        for c in range(4):
            axes[r, c].set_xlabel("theta index")
            axes[r, c].set_ylabel("f index")

    fig.colorbar(im1, ax=axes[:, :2].ravel().tolist(), shrink=0.75)
    fig.colorbar(im3, ax=axes[:, 2:].ravel().tolist(), shrink=0.75)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file", default=DATA_PATH)
    pa.add_argument("--out-dir", default=RUN_DIR)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--lambda-scale", type=float, default=0.1)
    pa.add_argument("--n-modes", type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    pa.add_argument("--scale-head-hidden", type=int, default=128)
    pa.add_argument("--field-checkpoint", default=None,
                    help="Checkpoint do FNO vencedor da semana passada. Pode ser FNO puro ou checkpoint fatorizado.")
    pa.add_argument("--freeze-field", action="store_true",
                    help="Congela o FNO do campo e treina apenas a rede da escala a.")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    config = {
        "model": {
            "type": "FNO_2D_factorized",
            "n_modes": args.n_modes,
            "hidden_channels": args.hidden_channels,
            "scale_head_hidden": args.scale_head_hidden,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "lambda_scale": args.lambda_scale,
            "field_loss": "relative_L2_pure",
            "scale_loss": "log_MSE",
            "field_checkpoint": args.field_checkpoint,
            "freeze_field": args.freeze_field,
        },
        "data": {"path": args.h5file},
    }
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    X, Y = load_snl_dataset(args.h5file)
    print(f"Dataset shape — X: {X.shape}  Y: {Y.shape}")

    Y_tilde, a = factorize_target(Y)

    n_total = len(X)
    n_train = int(0.8 * n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = TensorDataset(X[train_idx], Y_tilde[train_idx], a[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y_tilde[val_idx], a[val_idx], Y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = FactorizedSnlModel(
        n_modes=tuple(args.n_modes),
        hidden_channels=args.hidden_channels,
        scale_head_hidden=args.scale_head_hidden,
    ).to(device)

    if args.field_checkpoint is not None:
        load_field_checkpoint(model.field_model, args.field_checkpoint)

    if args.freeze_field:
        for p in model.field_model.parameters():
            p.requires_grad = False
        print("Branch do campo congelado: treinando apenas a rede da escala a.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_total, val_total = [], []
    train_field, val_field = [], []
    train_scale, val_scale = [], []
    val_phys = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_total = tr_field = tr_scale = 0.0

        for xb, yb_tilde, ab, yb_phys in train_loader:
            xb = xb.to(device)
            yb_tilde = yb_tilde.to(device)
            ab = ab.to(device)
            yb_phys = yb_phys.to(device)

            optimizer.zero_grad()
            pred_tilde, pred_a = model(xb)

            field_loss = relative_l2_loss(pred_tilde, yb_tilde)
            scale_loss = scale_log_mse(pred_a, ab)

            if args.freeze_field:
                loss = args.lambda_scale * scale_loss
            else:
                loss = field_loss + args.lambda_scale * scale_loss

            loss.backward()
            optimizer.step()

            tr_total += loss.item() * xb.size(0)
            tr_field += field_loss.item() * xb.size(0)
            tr_scale += scale_loss.item() * xb.size(0)

        tr_total /= len(train_loader.dataset)
        tr_field /= len(train_loader.dataset)
        tr_scale /= len(train_loader.dataset)

        model.eval()
        vl_total = vl_field = vl_scale = vl_phys = 0.0
        with torch.no_grad():
            for xb, yb_tilde, ab, yb_phys in val_loader:
                xb = xb.to(device)
                yb_tilde = yb_tilde.to(device)
                ab = ab.to(device)
                yb_phys = yb_phys.to(device)

                pred_tilde, pred_a = model(xb)
                field_loss = relative_l2_loss(pred_tilde, yb_tilde)
                scale_loss = scale_log_mse(pred_a, ab)
                total_loss = (args.lambda_scale * scale_loss) if args.freeze_field else (field_loss + args.lambda_scale * scale_loss)

                pred_phys = pred_tilde * pred_a.view(-1, 1, 1, 1)
                phys_loss = relative_l2_physical(pred_phys, yb_phys)

                vl_total += total_loss.item() * xb.size(0)
                vl_field += field_loss.item() * xb.size(0)
                vl_scale += scale_loss.item() * xb.size(0)
                vl_phys += phys_loss.item() * xb.size(0)

        vl_total /= len(val_loader.dataset)
        vl_field /= len(val_loader.dataset)
        vl_scale /= len(val_loader.dataset)
        vl_phys /= len(val_loader.dataset)

        train_total.append(tr_total); val_total.append(vl_total)
        train_field.append(tr_field); val_field.append(vl_field)
        train_scale.append(tr_scale); val_scale.append(vl_scale)
        val_phys.append(vl_phys)

        scheduler.step(vl_total)
        if vl_total < best_val:
            best_val = vl_total
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pth"))

        print(
            f"Epoch {epoch:03d} | "
            f"Train total: {tr_total:.4e} | Val total: {vl_total:.4e} | "
            f"Val field RelL2: {vl_field:.4e} | Val scale: {vl_scale:.4e} | "
            f"Val phys RelL2: {vl_phys:.4e}"
        )

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))

    factorized_params = {
        "mode": "factorized_scale",
        "eps": EPS,
        "description": "S_pred = a_pred * S_tilde_pred",
        "a_definition": "a = max_abs(S_nl) + eps",
        "field_target": "S_tilde = S_nl / a",
        "field_loss": "relative_L2_pure",
        "scale_loss": "log_MSE",
        "lambda_scale": args.lambda_scale,
        "field_checkpoint": args.field_checkpoint,
        "freeze_field": args.freeze_field,
        "a_train_mean": float(a[train_idx].mean()),
        "a_train_median": float(a[train_idx].median()),
    }
    with open(os.path.join(args.out_dir, "factorized_params.json"), "w") as f:
        json.dump(factorized_params, f, indent=2)

    metrics = {
        "final_train_total_loss": train_total[-1],
        "final_val_total_loss": val_total[-1],
        "final_val_field_loss": val_field[-1],
        "final_val_scale_loss": val_scale[-1],
        "final_val_physical_rel_l2": val_phys[-1],
        "best_val_total_loss": best_val,
        "train_stats": {
            "X": compute_stats(X[train_idx]),
            "Y": compute_stats(Y[train_idx]),
            "Y_tilde": compute_stats(Y_tilde[train_idx]),
            "a": compute_stats(a[train_idx]),
        },
        "config": config,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_curve(train_total, val_total, os.path.join(args.out_dir, "loss_total.pdf"), "Loss", "Total loss")
    save_curve(train_field, val_field, os.path.join(args.out_dir, "loss_field.pdf"), "Relative L2", "Field loss: Relative L2 puro")
    save_curve(train_scale, val_scale, os.path.join(args.out_dir, "loss_scale.pdf"), "Log-MSE", "Scale loss: log-MSE em a")

    plt.figure(figsize=(7, 4.5))
    plt.plot(val_phys, label="Validation physical Relative L2")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2")
    plt.title("Reconstruction error on physical S_nl")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "val_physical_rel_l2.pdf"), bbox_inches="tight")
    plt.close()

    # Exemplos de predição no conjunto de validação
    sample_indices = val_idx[: min(6, len(val_idx))].tolist()
    save_sample_predictions(
        model, X, Y, Y_tilde, a, sample_indices,
        os.path.join(args.out_dir, "samples_predictions.pdf"), device
    )

    print(f"\nArquivos salvos em: {args.out_dir}")
    print("Campo/shape treinado com Relative L2 puro, sem peak-weighted loss.")
    if args.freeze_field:
        print("O branch do campo foi congelado; apenas a rede de escala foi treinada.")


if __name__ == "__main__":
    main()

