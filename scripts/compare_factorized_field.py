#!/usr/bin/env python3
"""
Compara S_tilde_gt vs S_tilde_pred para o modelo fatorizado.

Compatível com run_training_factorized_scale.py corrigido:
  - shape/FNO com Relative L2 puro
  - amplitude a prevista separadamente

Saídas:
  - field_metrics_<split>.csv
  - field_rel_l2_hist_<split>.pdf
  - field_examples_<split>.pdf
  - field_summary_<split>.json
"""

import argparse
import csv
import json
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from fno_diffusion.model import make_fno_2d

EPS = 1e-8


class ScaleHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
            nn.Linear(64, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1), nn.Softplus(),
        )
    def forward(self, x):
        return self.net(x)


class FactorizedSnlModel(nn.Module):
    def __init__(self, n_modes=(16, 16), hidden_channels=64, scale_head_hidden=128):
        super().__init__()
        self.field_model = make_fno_2d(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1)
        self.scale_model = ScaleHead(hidden=scale_head_hidden)
    def forward(self, x):
        return self.field_model(x), self.scale_model(x)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint inválido.")
    cleaned = {}
    for k, v in state.items():
        if k == "_metadata":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_model(path, n_modes, hidden_channels, scale_head_hidden):
    model = FactorizedSnlModel(n_modes=n_modes, hidden_channels=hidden_channels, scale_head_hidden=scale_head_hidden)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    return model


def rel_l2(pred, target, eps=EPS):
    return float(np.linalg.norm((pred - target).ravel()) / (np.linalg.norm(target.ravel()) + eps))


def select_indices(n_total, split, max_samples, seed):
    n_train = int(0.8 * n_total)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    if split == "train":
        out = idx[:n_train]
    elif split == "val":
        out = idx[n_train:]
    else:
        out = torch.arange(n_total)
    if max_samples is not None:
        out = out[:max_samples]
    return out.tolist()


def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file", default="data/snl/snl_dataset.h5")
    pa.add_argument("--model", default="results_snl_factorized/model_best.pth")
    pa.add_argument("--out-dir", default="results_compare_factorized_field")
    pa.add_argument("--split", choices=["train", "val", "all"], default="val")
    pa.add_argument("--max-samples", type=int, default=None)
    pa.add_argument("--examples", type=int, default=6)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--n-modes", type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    pa.add_argument("--scale-head-hidden", type=int, default=128)
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.h5file, "r") as hf:
        X = torch.tensor(hf["X"][:], dtype=torch.float32).permute(0, 3, 1, 2)
        Y = torch.tensor(hf["Y"][:], dtype=torch.float32).permute(0, 3, 1, 2)

    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + EPS
    Y_tilde = Y / a

    indices = select_indices(len(X), args.split, args.max_samples, args.seed)
    model = load_model(args.model, tuple(args.n_modes), args.hidden_channels, args.scale_head_hidden)

    rows = []
    examples = []
    with torch.no_grad():
        for i in indices:
            pred_tilde, pred_a = model(X[i:i+1])
            gt = Y_tilde[i, 0].cpu().numpy()
            pr = pred_tilde[0, 0].cpu().numpy()
            er = pr - gt
            rows.append({
                "idx": i,
                "a_gt": float(a[i].item()),
                "a_pred": float(pred_a[0, 0].item()),
                "field_rel_l2": rel_l2(pr, gt),
                "field_mae": float(np.mean(np.abs(er))),
                "field_rmse": float(np.sqrt(np.mean(er ** 2))),
            })
            if len(examples) < args.examples:
                examples.append((i, gt, pr, er))

    csv_path = os.path.join(args.out_dir, f"field_metrics_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    rels = np.array([r["field_rel_l2"] for r in rows])

    plt.figure(figsize=(6, 4))
    plt.hist(rels, bins=30)
    plt.xlabel("Relative L2 em S_nl normalizado")
    plt.ylabel("Contagem")
    plt.title(f"Erro do campo normalizado ({args.split})")
    plt.tight_layout()
    hist_path = os.path.join(args.out_dir, f"field_rel_l2_hist_{args.split}.pdf")
    plt.savefig(hist_path, bbox_inches="tight"); plt.close()

    if examples:
        n = len(examples)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.0 * n))
        if n == 1:
            axes = np.array([axes])
        vmax = max(max(np.max(np.abs(gt)), np.max(np.abs(pr))) for _, gt, pr, _ in examples) + 1e-12
        for r, (idx, gt, pr, er) in enumerate(examples):
            im0 = axes[r, 0].imshow(gt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[r, 0].set_title(f"idx={idx} | S_tilde GT")
            im1 = axes[r, 1].imshow(pr, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[r, 1].set_title("S_tilde pred")
            im2 = axes[r, 2].imshow(er, origin="lower", aspect="auto", cmap="PiYG")
            axes[r, 2].set_title("Erro")
            for c in range(3):
                axes[r, c].set_xlabel("theta index")
                axes[r, c].set_ylabel("f index")
        fig.colorbar(im1, ax=axes[:, :2].ravel().tolist(), shrink=0.8)
        fig.colorbar(im2, ax=axes[:, 2].ravel().tolist(), shrink=0.8)
        plt.tight_layout()
        examples_path = os.path.join(args.out_dir, f"field_examples_{args.split}.pdf")
        plt.savefig(examples_path, bbox_inches="tight"); plt.close()
    else:
        examples_path = None

    summary = {
        "split": args.split,
        "n_samples": len(rows),
        "mean_field_rel_l2": float(np.mean(rels)),
        "median_field_rel_l2": float(np.median(rels)),
        "p90_field_rel_l2": float(np.percentile(rels, 90)),
        "worst_idx": int(rows[int(np.argmax(rels))]["idx"]),
        "best_idx": int(rows[int(np.argmin(rels))]["idx"]),
    }
    with open(os.path.join(args.out_dir, f"field_summary_{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Comparação S_tilde_gt vs S_tilde_pred ===")
    print(json.dumps(summary, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Hist: {hist_path}")
    if examples_path:
        print(f"Exemplos: {examples_path}")


if __name__ == "__main__":
    main()

