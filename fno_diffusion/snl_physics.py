# fno_diffusion/snl_physics.py

import numpy as np

def jonswap_spectrum(f, Hs, Tp, gamma, g=9.81):
    fp = 1.0 / Tp
    omega_p = 2 * np.pi * fp

    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-(f - fp)**2 / (2 * sigma**2 * fp**2))
    
    alpha_J = 0.076 * (Hs**2 * omega_p**4 / g**2)**0.22

    S = (
        alpha_J
        * g**2
        * (2*np.pi)**-4
        * f**-5
        * np.exp(-5/4 * (fp/f)**4)
        * gamma**r
    )

    return S

def directional_spreading(theta, theta0=0.0, s=20):
    delta_theta = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
    D = np.power(np.cos(0.5 * delta_theta), 2 * s)
    integral = np.trapz(D, theta)
    
    return D / integral


def generate_action_spectrum(omega, theta):
    Hs = np.random.uniform(1.0, 6.0)
    Tp = np.random.uniform(7.0, 15.0)
    gamma = np.random.uniform(1.0, 5.0)

    f = omega / (2*np.pi)

    S_f = jonswap_spectrum(f, Hs, Tp, gamma)
    S_w = S_f / (2*np.pi)

    D_theta = directional_spreading(theta)

    E = np.outer(S_w, D_theta)
    n = E / omega[:, None]

    n[n < 0] = 0.0
    return n, Tp


def angular_mean(n, theta):
    return np.trapz(n, theta, axis=1) / (2*np.pi)


def compute_psi(n, omega, theta, Tp):
    omega_p = 2*np.pi / Tp
    n_bar = angular_mean(n, theta)

    psi = ((omega[:, None] / omega_p)**24 * (n_bar[:, None]**2) * n)

    return psi

def laplacian_operator(psi, omega, theta):
    dtheta = theta[1] - theta[0]
    
    d2Psi_domega2 = np.gradient(np.gradient(psi, omega, axis=0), omega, axis=0)

    d2Psi_dtheta2 = (
        np.roll(psi, -1, axis=1)
        - 2 * psi
        + np.roll(psi, 1, axis=1)
    ) / dtheta**2

    L_Psi = 0.5 * d2Psi_domega2 + (1 / omega[:, None]**2) * d2Psi_dtheta2
    return L_Psi


def compute_snl_raw(n, omega, theta, Tp):
    psi = compute_psi(n, omega, theta, Tp)
    L_Psi = laplacian_operator(psi, omega, theta)
    return L_Psi / omega[:, None]**3


def compute_alpha(omega, theta, n_samples=100):
    max_val = 0.0

    for _ in range(n_samples):
        n, Tp = generate_action_spectrum(omega, theta)
        snl_raw = compute_snl_raw(n, omega, theta, Tp)
        max_val = max(max_val, np.max(np.abs(snl_raw)))

    return 1.0 / max_val


def sanity_checks(n, snl, omega, theta, tol=1e-2):
    integral = np.trapz(np.trapz(snl, theta, axis=1), omega)

    norm = np.trapz(np.trapz(np.abs(snl), theta, axis=1), omega)

    rel_error = abs(integral) / (norm + 1e-12)

    return rel_error
