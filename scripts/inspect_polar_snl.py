#!/usr/bin/env python3
"""
Ferramenta de Inspeção e Sanity Check do FNO treinado.

Modos de uso:
  1. Apenas ground truth (sem modelo):
       python scripts/inspect_polar_snl.py --idx 0

  2. Ground truth + predição do FNO (sanity check):
       python scripts/inspect_polar_snl.py --idx 0 --model results_snl/model.pth

  O modo 2 carrega automaticamente results_snl/norm_params.json para desfazer
  a normalização Y* -> Y antes de plotar.

  Layout modo 2 (2 linhas):
    Linha 0: Polar E(f,θ) | Polar Snl GT | Polar Snl FNO | Polar Erro
    Linha 1: 1D E(f)      | 1D Snl GT vs FNO (comparativo, ocupa 3 colunas)
"""
import os
import json
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Helper de plot polar
# ---------------------------------------------------------------------------

def _polar_plot(ax, theta, f, data, cmap, vmin=None, vmax=None, title="", clabel="",
               symlog=False, linthresh=None):
    """
    Plot polar com suporte a SymLogNorm para dados esparsos (quase tudo zero,
    com picos localizados) - caso típico do S_nl.

    Se symlog=True (ativado automaticamente para colormaps divergentes quando
    os dados têm amplitude muito pequena), usa escala logarítmica simétrica
    para tornar os picos visíveis mesmo quando a maioria dos valores é ~0.
    """
    TH, R = np.meshgrid(theta, f)

    if vmin is None: vmin = float(data.min())
    if vmax is None: vmax = float(data.max())

    norm = None
    if symlog:
        # linthresh: faixa linear ao redor do zero (evita log(0))
        # Se não fornecido, usa 1% do valor máximo absoluto
        lt = linthresh if linthresh is not None else max(abs(vmax), abs(vmin)) * 0.01
        lt = lt if lt > 0 else 1e-10
        norm = mcolors.SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax, base=10)

    pcm = ax.pcolormesh(TH, R, data, cmap=cmap, norm=norm,
                        vmin=(None if norm else vmin),
                        vmax=(None if norm else vmax),
                        shading="auto")
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
# Plot modo 1 — apenas ground truth (2x2)
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
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

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
# Plot modo 2 — ground truth + predição FNO + erro polar
# ---------------------------------------------------------------------------

def plot_comparison(idx, E_2d, Snl_gt, Snl_pred, f, theta, fp, Hs, gamma, th0, s):
    error        = Snl_pred - Snl_gt          # erro na escala física
    Snl_1d_gt   = _trapz(Snl_gt,   theta, axis=1)
    Snl_1d_pred = _trapz(Snl_pred, theta, axis=1)
    E_1d        = _trapz(E_2d,     theta, axis=1)

    # Layout: 2 linhas x 4 colunas
    fig = plt.figure(figsize=(22, 11))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(
        _build_title(idx, fp, Hs, gamma, th0, s) + "\n[Comparação: FNO vs Ground Truth]",
        fontsize=12, fontweight="bold"
    )

    vmax_snl = float(max(np.max(np.abs(Snl_gt)), np.max(np.abs(Snl_pred)))) or 1e-30
    vmax_err = float(np.max(np.abs(error))) or 1e-30

    # (0,0) Polar E - Input
    ax00 = fig.add_subplot(gs[0, 0], projection="polar")
    _polar_plot(ax00, theta, f, E_2d, "hot_r",
                title="E(f, θ) — Input", clabel="m² Hz⁻¹ rad⁻¹")

    # (0,1) Polar Snl GT
    ax01 = fig.add_subplot(gs[0, 1], projection="polar")
    _polar_plot(ax01, theta, f, Snl_gt, "RdBu_r",
                vmin=-vmax_snl, vmax=vmax_snl,
                title="S_nl — Ground Truth (DE3)",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

    # (0,2) Polar Snl FNO
    ax02 = fig.add_subplot(gs[0, 2], projection="polar")
    _polar_plot(ax02, theta, f, Snl_pred, "RdBu_r",
                vmin=-vmax_snl, vmax=vmax_snl,
                title="S_nl — Predição FNO",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

    # (0,3) Polar Erro (FNO − GT) - colormap divergente centrado em zero
    ax03 = fig.add_subplot(gs[0, 3], projection="polar")
    _polar_plot(ax03, theta, f, error, "PiYG",
                vmin=-vmax_err, vmax=vmax_err,
                title="Erro  (FNO − GT)",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

    # (1,0) 1D E(f)
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(f, E_1d, "k-", lw=2)
    ax10.axvline(fp, color="red", ls="--", lw=1, label=f"fp={fp:.2f} Hz")
    ax10.set_title("Espectro Integrado 1D  E(f)", fontsize=10)
    ax10.set_xlabel("f (Hz)"); ax10.set_ylabel("E(f) (m² Hz⁻¹)")
    ax10.grid(True, ls=":", alpha=0.6); ax10.legend()

    # (1,1:4) 1D Snl comparativo - ocupa as 3 colunas restantes
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
# Carregamento do modelo e parâmetros de normalização
# ---------------------------------------------------------------------------

def load_norm_params(model_path):
    """
    Carrega os parâmetros de normalização do norm_params.json.
    Suporta dois modos:
      - "per_sample": Y / (|Y|.max + eps)  - retorna (mode, eps, scale_fallback)
      - global z-score legacy               - retorna (mode, mu_Y, sigma_Y)
    """
    run_dir   = os.path.dirname(model_path)
    norm_path = os.path.join(run_dir, "norm_params.json")

    if not os.path.exists(norm_path):
        print(f"Aviso: '{norm_path}' não encontrado. Predição NÃO será desnormalizada.")
        return {"mode": "identity"}

    with open(norm_path) as f:
        params = json.load(f)

    mode = params.get("mode", "global")
    if mode == "per_sample":
        eps   = float(params.get("eps", 1e-8))
        scale = float(params.get("scale_train_mean", 1.0))
        print(f"Normalização por amostra carregada: eps={eps:.1e}, scale_fallback={scale:.4e}")
        return {"mode": "per_sample", "eps": eps, "scale_fallback": scale}
    else:
        mu_Y    = float(params["mu_Y"])
        sigma_Y = float(params["sigma_Y"])
        print(f"Normalização global carregada: mu_Y={mu_Y:.4e}, sigma_Y={sigma_Y:.4e}")
        return {"mode": "global", "mu_Y": mu_Y, "sigma_Y": sigma_Y}


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
    print(f"Modelo carregado: {model_path}")
    return model


def predict_snl(model, E_2d, Snl_gt, norm_params):
    """
    Roda o FNO e desfaz a normalização conforme o modo salvo em norm_params.

    Modos:
      "per_sample": a escala é calculada a partir do Snl_gt da própria amostra
                    (|Snl_gt|.max + eps), exatamente como foi feito no treino.
                    O scale_fallback só é usado se Snl_gt não estiver disponível.
      "global"    : pred_physical = pred_norm * sigma_Y + mu_Y
      "identity"  : sem desnormalização
    """
    import torch
    x = torch.tensor(E_2d[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred_norm = model(x)[0, 0].numpy()

    mode = norm_params.get("mode", "identity")
    if mode == "per_sample":
        eps = norm_params.get("eps", 1e-8)
        if Snl_gt is not None:
            # Escala real da amostra - idêntica à usada no treinamento
            scale = float(np.abs(Snl_gt).max()) + eps
        else:
            # Fallback: escala média do treino (menos preciso)
            scale = norm_params["scale_fallback"]
            print("Aviso: usando scale_fallback (Snl_gt não disponível)")
        return pred_norm * scale
    elif mode == "global":
        return pred_norm * norm_params["sigma_Y"] + norm_params["mu_Y"]
    else:
        return pred_norm


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
                    help="Caminho para model.pth — ativa modo comparação FNO vs GT")
    pa.add_argument("--n-modes",         type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    pa.add_argument("--out-dir", type=str, default="data/snl")
    args = pa.parse_args()

    if not os.path.exists(args.h5file):
        print(f"Erro: '{args.h5file}' não encontrado. Rode generate_snl_data.py primeiro.")
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

    # -- Modo 1: apenas ground truth -------------------------------
    if args.model is None:
        print(f"[Ground Truth] Amostra {idx}/{n_total-1}")
        fig = plot_ground_truth(idx, E_2d, Snl_gt, f, theta, fp, Hs, gamma, th0, s)
        out = os.path.join(args.out_dir, f"inspect_sample_{idx}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvo: {out}")
        return

    # -- Modo 2: comparação FNO vs GT ------------------------------
    print(f"[Comparação FNO] Amostra {idx}/{n_total-1}")
    norm_params = load_norm_params(args.model)
    model       = load_fno_model(args.model,
                                 n_modes=tuple(args.n_modes),
                                 hidden_channels=args.hidden_channels)
    Snl_pred    = predict_snl(model, E_2d, Snl_gt, norm_params)

    fig = plot_comparison(idx, E_2d, Snl_gt, Snl_pred,
                          f, theta, fp, Hs, gamma, th0, s)

    out = os.path.join(args.out_dir, f"comparison_sample_{idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()
