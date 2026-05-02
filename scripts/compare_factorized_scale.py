#!/usr/bin/env python3
"""
Compara a_gt vs a_pred para o modelo fatorizado.

Compatível com run_training_factorized_scale.py corrigido:
  - shape/FNO com Relative L2 puro
  - amplitude a prevista por FFN separada

Saídas:
  - scale_metrics_<split>.csv
  - scale_scatter_<split>.pdf
  - scale_rel_err_hist_<split>.pdf
  - scale_summary_<split>.json
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
    pa.add_argument("--out-dir", default="results_compare_factorized_scale")
    pa.add_argument("--split", choices=["train", "val", "all"], default="val")
    pa.add_argument("--max-samples", type=int, default=None)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--n-modes", type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    pa.add_argument("--scale-head-hidden", type=int, default=128)
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.h5file, "r") as hf:
        X = torch.tensor(hf["X"][:], dtype=torch.float32).permute(0, 3, 1, 2)
        Y = torch.tensor(hf["Y"][:], dtype=torch.float32).permute(0, 3, 1, 2)

    a_gt_all = Y.abs().amax(dim=(1, 2, 3)) + EPS
    indices = select_indices(len(X), args.split, args.max_samples, args.seed)
    model = load_model(args.model, tuple(args.n_modes), args.hidden_channels, args.scale_head_hidden)

    rows = []
    with torch.no_grad():
        for i in indices:
            _, a_pred = model(X[i:i+1])
            ag = float(a_gt_all[i].item())
            ap = float(a_pred[0, 0].item())
            rows.append({
                "idx": i,
                "a_gt": ag,
                "a_pred": ap,
                "abs_err": abs(ap - ag),
                "rel_err": abs(ap - ag) / (abs(ag) + EPS),
                "log_err": abs(np.log(ap + EPS) - np.log(ag + EPS)),
            })

    csv_path = os.path.join(args.out_dir, f"scale_metrics_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    a_gt = np.array([r["a_gt"] for r in rows])
    a_pred = np.array([r["a_pred"] for r in rows])
    rel = np.array([r["rel_err"] for r in rows])

    plt.figure(figsize=(5.3, 5.0))
    plt.scatter(a_gt, a_pred, s=12, alpha=0.7)
    mn = min(a_gt.min(), a_pred.min()); mx = max(a_gt.max(), a_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=1, label="ideal")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("a_gt")
    plt.ylabel("a_pred")
    plt.title(f"a_gt vs a_pred ({args.split})")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(args.out_dir, f"scale_scatter_{args.split}.pdf")
    plt.savefig(scatter_path, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(rel, bins=30)
    plt.xlabel("Erro relativo em a")
    plt.ylabel("Contagem")
    plt.title(f"Distribuição do erro relativo de a ({args.split})")
    plt.tight_layout()
    hist_path = os.path.join(args.out_dir, f"scale_rel_err_hist_{args.split}.pdf")
    plt.savefig(hist_path, bbox_inches="tight"); plt.close()

    summary = {
        "split": args.split,
        "n_samples": len(rows),
        "mean_rel_err": float(np.mean(rel)),
        "median_rel_err": float(np.median(rel)),
        "p90_rel_err": float(np.percentile(rel, 90)),
        "corr_log10": float(np.corrcoef(np.log10(a_gt + EPS), np.log10(a_pred + EPS))[0, 1]),
        "worst_idx": int(rows[int(np.argmax(rel))]["idx"]),
        "best_idx": int(rows[int(np.argmin(rel))]["idx"]),
    }
    with open(os.path.join(args.out_dir, f"scale_summary_{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Comparação a_gt vs a_pred ===")
    print(json.dumps(summary, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Scatter: {scatter_path}")
    print(f"Hist: {hist_path}")


if __name__ == "__main__":
    main()

