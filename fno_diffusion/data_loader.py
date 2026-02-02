# fno_diffusion/data_loader.py

import h5py
import torch


def load_pdebench_heat(path):
    """
    Load 1D diffusion–sorption dataset from PDEBench.

    Returns:
        u : (N, T, X)
        x : (X,)
        t : (T,)
    """
    u_list = []

    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())

        for k in keys:
            u = torch.tensor(f[k]["data"][:], dtype=torch.float32)
            u_list.append(u)

        u = torch.stack(u_list, dim=0)  # (N, T, X, 1)
        u = u.squeeze(-1)

        x = torch.tensor(f[keys[0]]["grid/x"][:], dtype=torch.float32)
        t = torch.tensor(f[keys[0]]["grid/t"][:], dtype=torch.float32)

    return u, x, t


def make_initial_final_pairs(u):
    """
    Create (u0, uT) pairs.
    """
    u0 = u[:, 0, :].unsqueeze(-1)
    uT = u[:, -1, :].unsqueeze(-1)
    return u0, uT
