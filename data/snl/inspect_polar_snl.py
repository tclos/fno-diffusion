#!/usr/bin/env python3
"""
Ferramenta para inspecionar os espectros gerados no arquivo .h5.
Gera um plot 2x2:
(Top) Polar E(f, theta) e S_nl(f, theta)
(Bot) Espectro 1D E(f) e Integral Direcional S_nl(f)
"""
import os, argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_trapz = getattr(np, "trapezoid", None) or np.trapz

def plot_sample(idx, E_2d, Snl_2d, f, theta, fp, Hs, gamma, th0, s):
    E_1d = _trapz(E_2d, theta, axis=1)
    Snl_1d = _trapz(Snl_2d, theta, axis=1)
    
    theta_deg = np.degrees(theta)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3)
    
    title = (f"Amostra {idx} | Hs={Hs:.2f}m, fp={fp:.2f}Hz, "
             f"gamma={gamma:.1f}, th0={np.degrees(th0):.0f}°, s={s}")
    if idx == 0:
        title += "  (J&P BENCHMARK)"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    TH, R = np.meshgrid(theta, f)
    
    # 1. Polar E(f, theta)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    pcm1 = ax1.pcolormesh(TH, R, E_2d, cmap='hot_r', shading='auto')
    ax1.set_title("Espectro E(f, θ)", pad=15)
    plt.colorbar(pcm1, ax=ax1, label='m² Hz⁻¹ rad⁻¹')
    
    # 2. Polar S_nl(f, theta)
    vmax = np.max(np.abs(Snl_2d))
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    pcm2 = ax2.pcolormesh(TH, R, Snl_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    ax2.set_title("S_nl(f, θ) - DE3", pad=15)
    plt.colorbar(pcm2, ax=ax2, label='m² Hz⁻¹ rad⁻¹ s⁻¹')
    
    # 3. 1D E(f)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(f, E_1d, 'k-', lw=2)
    ax3.axvline(fp, color='red', ls='--', lw=1, label=f'fp={fp:.2f}')
    ax3.set_title("Espectro Integrado 1D E(f)")
    ax3.set_xlabel("f (Hz)"); ax3.set_ylabel("E(f) (m² Hz⁻¹)")
    ax3.grid(True, linestyle=':', alpha=0.6); ax3.legend()
    
    # 4. 1D S_nl(f)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(f, Snl_1d, 'k-', lw=2)
    ax4.axhline(0, color='k', ls='-', lw=0.8)
    ax4.axvline(fp, color='red', ls='--', lw=1)
    ax4.set_title("Integral Direcional S_nl(f)")
    ax4.set_xlabel("f (Hz)"); ax4.set_ylabel("S_nl(f) (m² Hz⁻¹ s⁻¹)")
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    return fig

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("h5file", nargs="?", default="data/snl/snl_dataset.h5")
    pa.add_argument("--idx", type=int, default=0, help="Índice da amostra para plotar")
    args = pa.parse_args()

    if not os.path.exists(args.h5file):
        print(f"Erro: Arquivo {args.h5file} não encontrado. Rode generate_snl_data.py primeiro.")
        return

    with h5py.File(args.h5file, "r") as f_h5:
        idx = args.idx % f_h5["X"].shape[0]
        E_2d = f_h5["X"][idx, ..., 0]
        Snl_2d = f_h5["Y"][idx, ..., 0]
        f = f_h5["f"][:]
        theta = f_h5["theta"][:]
        
        fp = f_h5["fp"][idx]
        Hs = f_h5["Hs"][idx]  # <--- Corrigido aqui para ler Hs do dataset
        gamma = f_h5["gamma"][idx]
        th0 = f_h5["theta0"][idx]
        s = f_h5["s"][idx]

    fig = plot_sample(idx, E_2d, Snl_2d, f, theta, fp, Hs, gamma, th0, s)
    out_path = f"data/snl/inspect_sample_{idx}.png"
    os.makedirs("data/snl", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Salvo: {out_path}")

if __name__ == "__main__":
    main()
