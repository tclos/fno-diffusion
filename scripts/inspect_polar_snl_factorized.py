#!/usr/bin/env python3
"""
Inspeção do modelo fatorizado:
  E(f, theta) -> S_tilde(f, theta) e E(f, theta) -> a
  S_pred = a_pred * S_tilde_pred

Compatível com run_training_factorized_scale.py corrigido:
  - field/shape treinado com Relative L2 puro
  - scale model treinado separadamente com log-MSE em a
"""

import argparse
import json
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from fno_diffusion.model import make_fno_2d

_trapz = getattr(np, "trapezoid", None) or np.trapz
EPS = 1e-8


class ScaleHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64, hidden),
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
        self.field_model = make_fno_2d(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1)
        self.scale_model = ScaleHead(hidden=scale_head_hidden)
    def forward(self, x):
        return self.field_model(x), self.scale_model(x)


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


def load_model(model_path, n_modes=(16, 16), hidden_channels=64, scale_head_hidden=128):
    model = FactorizedSnlModel(n_modes=n_modes, hidden_channels=hidden_channels, scale_head_hidden=scale_head_hidden)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Modelo carregado: {model_path}")
    return model


def load_params(model_path):
    path = os.path.join(os.path.dirname(model_path), "factorized_params.json")
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"factorized_params.json carregado: {path}")
        print(f"field_loss = {params.get('field_loss', 'não especificado')}")
        print(f"freeze_field = {params.get('freeze_field', 'não especificado')}")
        return params
    print("Aviso: factorized_params.json não encontrado.")
    return {}


def rel_l2(pred, target, eps=EPS):
    return float(np.linalg.norm((pred - target).ravel()) / (np.linalg.norm(target.ravel()) + eps))


def polar_plot(ax, theta, f, data, title, cmap="RdBu_r", symmetric=True, symlog=True):
    TH, R = np.meshgrid(theta, f)
    if symmetric:
        vmax = float(np.max(np.abs(data))) or 1e-30
        vmin = -vmax
    else:
        vmin = float(np.min(data)); vmax = float(np.max(data))
    norm = None
    if symlog:
        norm = mcolors.SymLogNorm(linthresh=max(abs(vmin), abs(vmax)) * 0.01 + 1e-30, vmin=vmin, vmax=vmax, base=10)
    pcm = ax.pcolormesh(TH, R, data, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, shading="auto")
    ax.set_title(title, pad=14, fontsize=10)
    plt.colorbar(pcm, ax=ax, pad=0.08)


def build_title(idx, fp, Hs, gamma, th0, s):
    bench = " (J&P benchmark)" if idx == 0 else ""
    return f"Amostra {idx}{bench} | Hs={Hs:.2f} m, fp={fp:.3f} Hz, gamma={gamma:.2f}, theta0={np.degrees(th0):.1f} deg, s={s}"


def plot_comparison(idx, E, S_gt, S_pred, St_gt, St_pred, f, theta, fp, Hs, gamma, th0, s, a_gt, a_pred):
    err = S_pred - S_gt
    #E_1d = _trapz(E, theta, axis=1)
    S_gt_1d = _trapz(S_gt, theta, axis=1)
    S_pred_1d = _trapz(S_pred, theta, axis=1)
    rel_phys = rel_l2(S_pred, S_gt)
    rel_shape = rel_l2(St_pred, St_gt)
    rel_a = abs(a_pred - a_gt) / (abs(a_gt) + EPS)

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(build_title(idx, fp, Hs, gamma, th0, s) +
                 f"\na_gt={a_gt:.3e}, a_pred={a_pred:.3e}, err_a={rel_a:.3e}, shape RelL2={rel_shape:.3e}, physical RelL2={rel_phys:.3e}",
                 fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    polar_plot(ax, theta, f, E, "E(f, theta) input", cmap="hot_r", symmetric=False, symlog=False)
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    polar_plot(ax, theta, f, S_gt, "S_nl ground truth")
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    polar_plot(ax, theta, f, S_pred, "S_nl reconstruído")
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    polar_plot(ax, theta, f, err, "Erro físico (pred - GT)", cmap="PiYG")

    ax = fig.add_subplot(gs[1, 0], projection="polar")
    polar_plot(ax, theta, f, St_gt, "S_nl normalizado GT")
    ax = fig.add_subplot(gs[1, 1], projection="polar")
    polar_plot(ax, theta, f, St_pred, "S_nl normalizado pred")

    ax = fig.add_subplot(gs[1, 2:])
    # ax.plot(f, E_1d, "0.6", lw=1.4, label="E(f) input")
    ax2 = ax.twinx()
    ax2.plot(f, S_gt_1d, "k-", lw=2, label="S_nl GT")
    ax2.plot(f, S_pred_1d, "r--", lw=1.8, label="S_nl pred")
    ax.axvline(fp, color="blue", ls="--", lw=1, label=f"fp={fp:.3f} Hz")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("E(f)")
    ax2.set_ylabel("Integral direcional de S_nl")
    ax.grid(True, ls=":", alpha=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.set_title("Integral direcional")
    return fig, {"rel_phys": rel_phys, "rel_shape": rel_shape, "rel_a": rel_a}


def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("h5file", nargs="?", default="data/snl/snl_dataset.h5")
    pa.add_argument("--idx", type=int, default=0)
    pa.add_argument("--model", default="results_snl_factorized/model_best.pth")
    pa.add_argument("--out-dir", default="data/snl")
    pa.add_argument("--n-modes", type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    pa.add_argument("--scale-head-hidden", type=int, default=128)
    args = pa.parse_args()

    with h5py.File(args.h5file, "r") as hf:
        n_total = hf["X"].shape[0]
        idx = args.idx % n_total
        E = hf["X"][idx, ..., 0]
        S_gt = hf["Y"][idx, ..., 0]
        f = hf["f"][:]
        theta = hf["theta"][:]
        fp = float(hf["fp"][idx]); Hs = float(hf["Hs"][idx]); gamma = float(hf["gamma"][idx]); th0 = float(hf["theta0"][idx]); s = int(hf["s"][idx])

    load_params(args.model)
    model = load_model(args.model, tuple(args.n_modes), args.hidden_channels, args.scale_head_hidden)

    x = torch.tensor(E[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        St_pred_t, a_pred_t = model(x)
    St_pred = St_pred_t[0, 0].cpu().numpy()
    a_pred = float(a_pred_t[0, 0].cpu().item())

    a_gt = float(np.max(np.abs(S_gt)) + EPS)
    St_gt = S_gt / a_gt
    S_pred = St_pred * a_pred

    print(f"a_gt={a_gt:.6e} | a_pred={a_pred:.6e} | rel_a={abs(a_pred-a_gt)/(abs(a_gt)+EPS):.6e}")
    print(f"shape RelL2={rel_l2(St_pred, St_gt):.6e} | physical RelL2={rel_l2(S_pred, S_gt):.6e}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, _ = plot_comparison(idx, E, S_gt, S_pred, St_gt, St_pred, f, theta, fp, Hs, gamma, th0, s, a_gt, a_pred)
    out = os.path.join(args.out_dir, f"factorized_comparison_sample_{idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()

