"""Mikro şebeke yük-frekans kontrolü için doğrusal dinamik model (SPEC §2, §3.2).

Durum sırası STATE_NAMES, giriş sırası [u, d]. Ayrıklaştırma sıfırıncı mertebe
tutucu (ZOH) ile kesin yapılır: artırılmış matrisin üstelinden.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from config import BessParams, GridParams

STATE_NAMES = ("delta_f_pu", "dp_g", "dp_m", "dp_c", "p_ess")


def build_continuous_model(grid: GridParams, bess: BessParams) -> tuple[np.ndarray, np.ndarray]:
    """Sürekli zaman matrisleri A (5x5) ve B (5x2); giriş v = [u, d]."""
    two_h = 2.0 * grid.H
    a = np.zeros((5, 5), dtype=float)
    b = np.zeros((5, 2), dtype=float)

    # Δf: salınım denklemi
    a[0, 0] = -grid.D / two_h
    a[0, 2] = 1.0 / two_h
    a[0, 4] = 1.0 / two_h
    b[0, 1] = 1.0 / two_h
    # ΔP_g: governor
    a[1, 0] = -1.0 / (grid.R * grid.T_g)
    a[1, 1] = -1.0 / grid.T_g
    a[1, 3] = 1.0 / grid.T_g
    # ΔP_m: türbin
    a[2, 1] = 1.0 / grid.T_t
    a[2, 2] = -1.0 / grid.T_t
    # ΔP_c: AGC (batarya boşaltma terimi ile)
    a[3, 0] = -grid.K_i_agc * grid.beta
    a[3, 4] = grid.K_i_agc
    # P_ess: dönüştürücü gecikmesi
    a[4, 4] = -1.0 / bess.T_ess
    b[4, 0] = 1.0 / bess.T_ess
    return a, b


def build_discrete_model(grid: GridParams, bess: BessParams) -> tuple[np.ndarray, np.ndarray]:
    """ZOH ile kesin ayrıklaştırılmış Ad (5x5), Bd (5x2), adım = grid.dt."""
    a, b = build_continuous_model(grid, bess)
    n, m = b.shape
    aug = np.zeros((n + m, n + m), dtype=float)
    aug[:n, :n] = a
    aug[:n, n:] = b
    e = expm(aug * grid.dt)
    return e[:n, :n].copy(), e[:n, n:].copy()
