# fno_diffusion/snl_physics.py
#
# Motor Físico para geração do termo não-linear S_nl (Modelo DE3)
# Incorpora estabilidade numérica para o termo w^24 e geração baseada em Hs.

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
g = 9.81


def generate_jonswap_2d(f, theta, fp=0.4, gamma=3.3, theta0=0.0, s=1, Hs=None, alpha=0.0081, jp_benchmark=False):
    """
    Gera o Espectro de Energia 2D E(f, theta) baseado em JONSWAP.
    Se Hs for fornecido, a energia total é reescalada para bater exatamente com a Altura Significativa.
    """
    f_grid, theta_grid = np.meshgrid(f, theta, indexing='ij')
    
    # JONSWAP base
    sigma_j = np.where(f_grid <= fp, 0.07, 0.09)
    PM = alpha * g**2 * (2.0*np.pi)**(-4) * f_grid**(-5) * np.exp(-1.25 * (fp/f_grid)**4)
    J = PM * gamma**(np.exp(-0.5 * ((f_grid - fp) / (sigma_j * fp))**2))
    
    if jp_benchmark:
        # Espalhamento direcional do Benchmark J&P: (2/pi) cos^2(theta) para |theta| <= pi/2
        delta = (theta_grid - theta0 + np.pi) % (2*np.pi) - np.pi
        mask = np.abs(delta) <= np.pi/2.0
        D_theta = np.where(mask, (2.0 / np.pi) * np.cos(delta)**2, 0.0)
    else:
        # Espalhamento direcional genérico: cos^{2s}((theta-theta0)/2)
        delta = (theta_grid - theta0 + np.pi) % (2*np.pi) - np.pi
        D_unnorm = np.maximum(np.cos(delta / 2.0), 0.0)**(2*s)
        norm = _trapz(D_unnorm[0, :], theta)
        D_theta = D_unnorm / norm if norm > 0 else np.ones_like(theta_grid)/(2*np.pi)
        
    Ef_2d = J * D_theta
    
    # Reescala a energia se um Hs alvo for estipulado (Ondas Oceânicas)
    if Hs is not None:
        E_1d = _trapz(Ef_2d, theta, axis=1)
        m0 = _trapz(E_1d, f)
        if m0 > 0:
            Hs_atual = 4.0 * np.sqrt(m0)
            Ef_2d = Ef_2d * (Hs / Hs_atual)**2
            
    return Ef_2d


def compute_de3_snl(Ef_2d, f, theta, alpha3=2.6e-7):
    """
    Calcula o S_nl(f, theta) pelo modelo DE3 (Pushkarev et al. 2004).
    Implementa a correção numérica de (omega/omega_p)^24 sugerida no plano teórico.
    """
    # Usando float64 internamente para evitar estouro nas derivadas
    f_grid, theta_grid = np.meshgrid(f, theta, indexing='ij')
    omega_1d = (2.0 * np.pi * f).astype(np.float64)
    omega_grid = (2.0 * np.pi * f_grid).astype(np.float64)
    Ef_2d = Ef_2d.astype(np.float64)
    
    # 1. Converter Espectro para Densidade de Ação n(ω, θ)
    n_omega_theta = Ef_2d * g**2 / (4.0 * np.pi * omega_grid**4)
    
    # 2. Encontrar omega_p para ancoragem numérica
    E_1d = _trapz(Ef_2d, theta, axis=1)
    idx_p = np.argmax(E_1d)
    omega_p = omega_1d[idx_p] if E_1d[idx_p] > 0 else 1.0
    
    # 3. D = <n>²
    n_mean = _trapz(n_omega_theta, theta, axis=1) / (2.0 * np.pi)
    D_coeff = n_mean**2
    D_grid = D_coeff[:, np.newaxis] * np.ones_like(theta_grid)
    
    # 4. Termo Estabilizado Psi = (ω / ω_p)^24 * D * n
    Psi = (omega_grid / omega_p)**24 * D_grid * n_omega_theta
    
    # 5. Operador Diferencial em Psi
    d_Psi_dw = np.gradient(Psi, omega_1d, axis=0)
    d2_Psi_dw2 = np.gradient(d_Psi_dw, omega_1d, axis=0)
    
    d_Psi_dth = np.gradient(Psi, theta, axis=1)
    d2_Psi_dth2 = np.gradient(d_Psi_dth, theta, axis=1)
    
    L_Psi = 0.5 * d2_Psi_dw2 + (1.0 / omega_grid**2) * d2_Psi_dth2
    
    # 6. ∂n/∂t = α3 * (ω_p^24 / ω^3) * L[Psi]
    # A matemática é idêntica à formulação original, mas imune a overflow de float
    dn_dt = alpha3 * (omega_p**24 / omega_grid**3) * L_Psi
    
    # 7. S_nl(f,θ) = ∂E(f,θ)/∂t
    Snl_f2d = dn_dt * (4.0 * np.pi * omega_grid**4) / g**2
    
    return Snl_f2d.astype(np.float32)
