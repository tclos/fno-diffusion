# scripts/generate_snl_data.py

import argparse
import os
import h5py
import numpy as np

from fno_diffusion.snl_physics import (
    generate_action_spectrum,
    compute_snl_raw,
    compute_alpha,
    sanity_checks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SNL dataset")

    parser.add_argument("--n-samples", type=int, default=10000,
                        help="")

    parser.add_argument("--n-omega", type=int, default=64,
                        help="")

    parser.add_argument("--n-theta", type=int, default=64,
                        help="")

    parser.add_argument("--omega-min", type=float, default=0.3,
                        help="")

    parser.add_argument("--omega-max", type=float, default=3.0,
                        help="")

    parser.add_argument("--alpha-samples", type=int, default=100,
                        help="")

    parser.add_argument("--seed", type=int, default=42,
                        help="")

    parser.add_argument("--out", type=str, default="snl_dataset.h5",
                        help="")

    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    theta = np.linspace(0.0, 2*np.pi, args.n_theta, endpoint=False)

    print("Grid:")
    print(f"  omega  : {args.n_omega} points [{omega[0]:.2f}, {omega[-1]:.2f}]")
    print(f"  theta  : {args.n_theta} points [0, 2*pi)")
    print()

    print("Calibrating alpha...")
    alpha = compute_alpha(
        omega,
        theta,
        n_samples=args.alpha_samples
    )
    print(f"  alpha = {alpha:.3e}")
    print()

    X = np.zeros((args.n_samples, args.n_omega, args.n_theta, 1), dtype=np.float32)
    Y = np.zeros_like(X)

    conservation_errors = []

    print("Generating samples...")
    for i in range(args.n_samples):
        n, Tp = generate_action_spectrum(omega, theta)
        snl_raw = compute_snl_raw(n, omega, theta, Tp)
        snl = alpha * snl_raw

        rel_error = sanity_checks(n, snl, omega, theta)
        conservation_errors.append(rel_error)

        X[i, ..., 0] = n.astype(np.float32)
        Y[i, ..., 0] = snl.astype(np.float32)

        if (i + 1) % max(1, args.n_samples // 10) == 0:
            print(f"  {i+1}/{args.n_samples} samples generated")

    conservation_errors = np.array(conservation_errors)

    print()
    print("Conservation check:")
    print(f"  mean error : {conservation_errors.mean():.3e}")
    print(f"  max  error : {conservation_errors.max():.3e}")
    print()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Saving dataset to {args.out}")
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("Y", data=Y, compression="gzip")

        f.create_dataset("omega", data=omega)
        f.create_dataset("theta", data=theta)

        f.attrs["alpha"] = alpha
        f.attrs["n_samples"] = args.n_samples
        f.attrs["n_omega"] = args.n_omega
        f.attrs["n_theta"] = args.n_theta
        f.attrs["omega_min"] = args.omega_min
        f.attrs["omega_max"] = args.omega_max
        f.attrs["seed"] = args.seed

    print("Done")


if __name__ == "__main__":
    main()
