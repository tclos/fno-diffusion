#!/usr/bin/env python
"""
Gerador de Dataset para o FNO: Espectro E(f, theta) -> DE3 S_nl(f, theta).
A Amostra 0 é rigidamente o Benchmark de J&P para calibração.
As amostras subsequentes simulam ondas oceânicas com Hs de 1 a 6 metros.
"""
import argparse, os, time
import numpy as np
import h5py
from fno_diffusion.snl_physics import generate_jonswap_2d, compute_de3_snl, _trapz

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--n-samples",  type=int,   default=1000, help="Total de amostras")
    pa.add_argument("--n-f",        type=int,   default=128,  help="Resolução em frequência (Potência de 2 é ideal pro FNO)")
    pa.add_argument("--n-theta",    type=int,   default=64,   help="Resolução direcional")
    pa.add_argument("--f-min",      type=float, default=0.04, help="Frequência mínima (Hz)")
    pa.add_argument("--f-max",      type=float, default=1.0,  help="Frequência máxima (Hz)")
    pa.add_argument("--seed",       type=int,   default=42)
    pa.add_argument("--out",        type=str,   default="data/snl/snl_dataset.h5")
    args = pa.parse_args()

    np.random.seed(args.seed)
    
    f = np.linspace(args.f_min, args.f_max, args.n_f)
    theta = np.linspace(-np.pi, np.pi, args.n_theta)
    
    Ns, Nf, Nt = args.n_samples, args.n_f, args.n_theta
    X = np.empty((Ns, Nf, Nt, 1), np.float32)  # E(f, theta)
    Y = np.empty((Ns, Nf, Nt, 1), np.float32)  # Snl(f, theta)
    
    # Metadata
    fp_a = np.empty(Ns, np.float32); hs_a = np.empty(Ns, np.float32)
    gamma_a = np.empty(Ns, np.float32); th0_a = np.empty(Ns, np.float32)
    s_a = np.empty(Ns, np.int32)
    
    t0 = time.time()
    print(f"Gerando {Ns} amostras. Amostra 0 = Benchmark J&P. Demais = Oceano Real.")
    
    global_scale = 1.0
    
    for i in range(Ns):
        if i == 0:
            # Benchmark J&P exato
            fp_i, Hs_i, gamma_i, th0_i, s_i = 0.4, None, 3.3, 0.0, 1
            jp_bench = True
        else:
            # Ondas de Oceano (Períodos de 5 a 15 segundos, Hs de 1 a 6 metros)
            Tp_i = np.random.uniform(5.0, 15.0)
            fp_i = 1.0 / Tp_i
            Hs_i = np.random.uniform(1.0, 6.0)
            gamma_i = np.random.uniform(1.0, 5.0)
            th0_i = np.random.uniform(-np.pi, np.pi)
            s_i = np.random.randint(1, 15)
            jp_bench = False
            
        Ef_2d = generate_jonswap_2d(f, theta, fp=fp_i, gamma=gamma_i, theta0=th0_i, s=s_i, Hs=Hs_i, jp_benchmark=jp_bench)
        Snl_f2d = compute_de3_snl(Ef_2d, f, theta)
        
        if i == 0:
            # Calibração baseada no Benchmark (-4.0e-5 de trough)
            Snl_f1d = _trapz(Snl_f2d, theta, axis=1)
            target_trough = -4.0e-5
            raw_min = np.min(Snl_f1d)
            global_scale = target_trough / raw_min if raw_min < 0 else 1.0
            print(f"  -> Fator de Escala Global calibrado no J&P: {global_scale:.4e}")
            
        Snl_f2d_scaled = Snl_f2d * global_scale
        
        X[i, :, :, 0] = Ef_2d
        Y[i, :, :, 0] = Snl_f2d_scaled
        
        # Registrar o Hs real no array de metadados
        E_1d = _trapz(Ef_2d, theta, axis=1)
        Hs_real = 4.0 * np.sqrt(max(_trapz(E_1d, f), 0.0))
        
        fp_a[i] = fp_i; hs_a[i] = Hs_real; gamma_a[i] = gamma_i
        th0_a[i] = th0_i; s_a[i] = s_i
        
        if (i+1) % max(1, Ns//10) == 0:
            print(f"  [{i+1:>{len(str(Ns))}}/{Ns}]  decorrido {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.out) or "data", exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("X", data=X, compression="gzip")
        hf.create_dataset("Y", data=Y, compression="gzip")
        hf.create_dataset("f", data=f); hf.create_dataset("theta", data=theta)
        hf.create_dataset("fp", data=fp_a); hf.create_dataset("Hs", data=hs_a)
        hf.create_dataset("gamma", data=gamma_a); hf.create_dataset("theta0", data=th0_a)
        hf.create_dataset("s", data=s_a)
        hf.attrs["scale"] = global_scale
        hf.attrs["physics"] = "DE3 (Pushkarev 2004) - Input E(f, theta) - Ocean Swells"
        
    print(f"\nSalvo em {args.out} ({Ns} amostras geradas)")

if __name__ == "__main__":
    main()
