#!/usr/bin/env python3
"""
Ferramenta para inspecionar os espectros gerados no arquivo .h5.

Modos de uso:
  1. Apenas ground truth (sem modelo):
       python scripts/inspect_polar_snl.py --idx 0

  2. Ground truth + predição do FNO (sanity check):
       python scripts/inspect_polar_snl.py --idx 0 --model results_snl/model.pth
"""
import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polar_plot(ax, theta, f, data, cmap, vmin=None, vmax=None, title="", clabel=""):
    TH, R = np.meshgrid(theta, f)
    pcm = ax.pcolormesh(TH, R, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    plt.colorbar(pcm, ax=ax, label=clabel, pad=0.08)
    ax.set_title(title, pad=15, fontsize=10)
    return pcm


def _build_title(idx, fp, Hs, gamma, th0, s):
    tag = "  (J&P BENCHMARK)" if idx == 0 else ""
    return (
        f"Amostra {idx}{tag} | "
        f"Hs={Hs:.2f} m, fp={fp:.2f} Hz, "
        f"γ={gamma:.1f}, θ₀={np.degrees(th0):.0f}°, s={s}"
    )


# ---------------------------------------------------------------------------
# Plot modo 1: apenas ground truth (2x2)
# ---------------------------------------------------------------------------

def plot_ground_truth(idx, E_2d, Snl_2d, f, theta, fp, Hs, gamma, th0, s):
    E_1d   = _trapz(E_2d,   theta, axis=1)
    Snl_1d = _trapz(Snl_2d, theta, axis=1)

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4)
    fig.suptitle(_build_title(idx, fp, Hs, gamma, th0, s), fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    _polar_plot(ax1, theta, f, E_2d, "hot_r",
                title="Espectro E(f, θ)", clabel="m² Hz⁻¹ rad⁻¹")

    vmax = float(np.max(np.abs(Snl_2d))) or 1e-30
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    _polar_plot(ax2, theta, f, Snl_2d, "RdBu_r",
                vmin=-vmax, vmax=vmax,
                title="S_nl(f, θ) — DE3 (Ground Truth)",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(f, E_1d, "k-", lw=2)
    ax3.axvline(fp, color="red", ls="--", lw=1, label=f"fp={fp:.2f} Hz")
    ax3.set_title("Espectro Integrado 1D  E(f)", fontsize=10)
    ax3.set_xlabel("f (Hz)"); ax3.set_ylabel("E(f) (m² Hz⁻¹)")
    ax3.grid(True, ls=":", alpha=0.6); ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(f, Snl_1d, "k-", lw=2)
    ax4.axhline(0, color="k", ls="-", lw=0.8)
    ax4.axvline(fp, color="red", ls="--", lw=1)
    ax4.set_title("Integral Direcional  S_nl(f)", fontsize=10)
    ax4.set_xlabel("f (Hz)"); ax4.set_ylabel("S_nl(f) (m² Hz⁻¹ s⁻¹)")
    ax4.grid(True, ls=":", alpha=0.6)

    return fig


# ---------------------------------------------------------------------------
# Plot modo 2: ground truth + predição FNO
# ---------------------------------------------------------------------------

def plot_comparison(idx, E_2d, Snl_gt, Snl_pred, f, theta, fp, Hs, gamma, th0, s):
    Snl_1d_gt   = _trapz(Snl_gt,   theta, axis=1)
    Snl_1d_pred = _trapz(Snl_pred, theta, axis=1)
    E_1d        = _trapz(E_2d,     theta, axis=1)

    fig = plt.figure(figsize=(18, 11))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(
        _build_title(idx, fp, Hs, gamma, th0, s) + "\n[Comparação: FNO vs Ground Truth]",
        fontsize=12, fontweight="bold"
    )
    
    vmax_snl_gt = float(np.max(np.abs(Snl_gt))) or 1e-30
    vmax_snl = float(np.max(np.abs(Snl_pred))) or 1e-30

    ax00 = fig.add_subplot(gs[0, 0], projection="polar")
    _polar_plot(ax00, theta, f, E_2d, "hot_r",
                title="E(f, θ) — Input", clabel="m² Hz⁻¹ rad⁻¹")

    ax01 = fig.add_subplot(gs[0, 1], projection="polar")
    _polar_plot(ax01, theta, f, Snl_gt, "RdBu_r",
                vmin=-vmax_snl_gt, vmax=vmax_snl_gt,
                title="S_nl — Ground Truth (DE3)",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹")

    ax02 = fig.add_subplot(gs[0, 2], projection="polar")
    _polar_plot(ax02, theta, f, Snl_pred, "RdBu_r",
                vmin=-vmax_snl, vmax=vmax_snl,
                title="S_nl — Predição FNO",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹")

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(f, E_1d, "k-", lw=2)
    ax10.axvline(fp, color="red", ls="--", lw=1, label=f"fp={fp:.2f} Hz")
    ax10.set_title("Espectro Integrado 1D  E(f)", fontsize=10)
    ax10.set_xlabel("f (Hz)"); ax10.set_ylabel("E(f) (m² Hz⁻¹)")
    ax10.grid(True, ls=":", alpha=0.6); ax10.legend()

    ax11 = fig.add_subplot(gs[1, 1:])
    ax11.plot(f, Snl_1d_gt,   "k-",  lw=2.0, label="Ground Truth (DE3)")
    ax11.plot(f, Snl_1d_pred, "r--", lw=1.8, label="FNO predito")
    ax11.axhline(0,  color="k",    ls="-",  lw=0.7)
    ax11.axvline(fp, color="blue", ls="--", lw=1.0, label=f"fp={fp:.2f} Hz")
    ax11.set_title("Integral Direcional  S_nl(f)", fontsize=10)
    ax11.set_xlabel("f (Hz)"); ax11.set_ylabel("S_nl(f) (m² Hz⁻¹ s⁻¹)")
    ax11.legend(fontsize=9); ax11.grid(True, ls=":", alpha=0.6)

    return fig


# ---------------------------------------------------------------------------
# Carregamento do FNO
# ---------------------------------------------------------------------------

def load_fno_model(model_path, n_modes=(16, 16), hidden_channels=64):
    import torch
    from fno_diffusion.model import make_fno_2d

    model = make_fno_2d(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


def predict_snl(model, E_2d):
    import torch
    x = torch.tensor(E_2d[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)
    return pred[0, 0].numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(
        description="Inspeção e Sanity Check do FNO treinado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pa.add_argument("h5file",  nargs="?", default="data/snl/snl_dataset.h5")
    pa.add_argument("--idx",   type=int,  default=0,
                    help="Índice da amostra (0 = Benchmark J&P)")
    pa.add_argument("--model", type=str,  default=None,
                    help="Caminho para model.pth: ativa modo comparação FNO vs GT")
    pa.add_argument("--n-modes",         type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    pa.add_argument("--out-dir", type=str, default="data/snl")
    args = pa.parse_args()

    if not os.path.exists(args.h5file):
        print(f"Error: '{args.h5file}' not found. Run generate_snl_data.py first.")
        return

    with h5py.File(args.h5file, "r") as hf:
        n_total = hf["X"].shape[0]
        idx    = args.idx % n_total
        E_2d   = hf["X"][idx, ..., 0]
        Snl_gt = hf["Y"][idx, ..., 0]
        f      = hf["f"][:]
        theta  = hf["theta"][:]
        fp     = float(hf["fp"][idx])
        Hs     = float(hf["Hs"][idx])
        gamma  = float(hf["gamma"][idx])
        th0    = float(hf["theta0"][idx])
        s      = int(hf["s"][idx])

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model is None:
        print(f"[Ground Truth] Sample {idx}/{n_total-1}")
        fig = plot_ground_truth(idx, E_2d, Snl_gt, f, theta, fp, Hs, gamma, th0, s)
        out = os.path.join(args.out_dir, f"inspect_sample_{idx}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvo: {out}")
        return

    print(f"[Comparação FNO] Amostra {idx}/{n_total-1}")
    model    = load_fno_model(args.model,
                              n_modes=tuple(args.n_modes),
                              hidden_channels=args.hidden_channels)
    Snl_pred = predict_snl(model, E_2d)

    fig = plot_comparison(idx, E_2d, Snl_gt, Snl_pred,
                          f, theta, fp, Hs, gamma, th0, s)

    out = os.path.join(args.out_dir, f"comparison_sample_{idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
