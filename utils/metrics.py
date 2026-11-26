"""Metrikler ve istatistik (SPEC 3.8 ve 7).

Arayuz:
    compute_metrics(traj, scenario=None, grid=GridParams(), bess=BessParams(), settle_band_hz=0.02)
    aggregate(values, n_boot=2000, seed=0)
    paired_test(a, b)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats

from config import BessParams, EvalParams, GridParams, Scenario

_MHZ_PER_HZ = 1000.0      # Hz -> mHz
_KW_PER_MW = 1000.0       # MW -> kW
_S_PER_H = 3600.0         # s -> h
_BAND_LEVELS_MHZ = (50, 100, 150)   # bant yuzdesi metrikleri icin esikler [mHz]
_CI_PERCENT = (2.5, 97.5)           # bootstrap %95 araligi


def _first_load_step_window(scenario: Scenario | None, n: int) -> tuple[int, int] | None:
    """Ilk yuk adiminin penceresi [baslangic, bitis). Adim yoksa None."""
    if scenario is None:
        return None
    steps = scenario.meta.get("load_steps") if scenario.meta else None
    if not steps:
        return None
    idx = sorted(int(s[0]) for s in steps)
    start = idx[0]
    if start >= n:
        return None
    later = [i for i in idx if i > start]
    end = min(later[0], n) if later else n
    return start, end


def _settling_time(df: np.ndarray, scenario: Scenario | None, dt: float, band_hz: float) -> float:
    """Ilk yuk adimi icin yerlesme suresi [s]; tanim SPEC 7."""
    window = _first_load_step_window(scenario, len(df))
    if window is None:
        return math.nan
    start, end = window
    outside = np.flatnonzero(np.abs(df[start:end]) >= band_hz)
    if outside.size == 0:
        return 0.0
    last = int(outside[-1])
    if last == end - start - 1:      # pencere sonunda hala bant disinda
        return math.nan
    return float((last + 1) * dt)


def compute_metrics(traj: dict[str, np.ndarray], scenario: Scenario | None = None,
                    grid: GridParams = GridParams(), bess: BessParams = BessParams(),
                    settle_band_hz: float = EvalParams.settle_band_hz) -> dict[str, float]:
    """Bir bolum yorungesinden SPEC 7'deki tum metrikleri hesaplar (hepsi Python float)."""
    dt = grid.dt
    df = np.asarray(traj["delta_f_hz"], dtype=float)
    p_ess = np.asarray(traj["p_ess"], dtype=float)
    u_ess = np.asarray(traj["u_ess"], dtype=float)
    soc = np.asarray(traj["soc"], dtype=float)
    p_diesel = np.asarray(traj["p_diesel"], dtype=float)
    shield = np.asarray(traj["shield_active"], dtype=float)
    reward = np.asarray(traj["reward"], dtype=float)

    abs_df = np.abs(df)
    rocof = np.abs(np.diff(df)) / dt
    throughput_kwh = float(np.sum(np.abs(p_ess)) * grid.s_base_mw * _KW_PER_MW * dt / _S_PER_H)

    m: dict[str, float] = {
        "rmse_mhz": math.sqrt(float(np.mean(df ** 2))) * _MHZ_PER_HZ,
        "mae_mhz": float(np.mean(abs_df)) * _MHZ_PER_HZ,
        "max_dev_mhz": float(np.max(abs_df)) * _MHZ_PER_HZ,
        "nadir_mhz": float(np.min(df)) * _MHZ_PER_HZ,
        "zenith_mhz": float(np.max(df)) * _MHZ_PER_HZ,
    }
    for level in _BAND_LEVELS_MHZ:
        m[f"band_{level}_pct"] = 100.0 * float(np.mean(abs_df < level / _MHZ_PER_HZ))
    m["rocof_max_hz_s"] = float(np.max(rocof)) if rocof.size else 0.0
    m["rocof_mean_hz_s"] = float(np.mean(rocof)) if rocof.size else 0.0
    m["settling_time_s"] = _settling_time(df, scenario, dt, settle_band_hz)
    m["bess_throughput_kwh"] = throughput_kwh
    m["bess_efc"] = throughput_kwh / (2.0 * bess.e_mwh * _KW_PER_MW)
    m["soc_min"] = float(np.min(soc))
    m["soc_max"] = float(np.max(soc))
    m["soc_final"] = float(soc[-1])
    m["diesel_travel_pu"] = float(np.sum(np.abs(np.diff(p_diesel))))
    m["control_jitter"] = float(np.sum(np.abs(np.diff(u_ess)))) / bess.p_max
    m["autonomy_pct"] = 100.0 * (1.0 - float(np.mean(shield)))
    m["mean_cost"] = -float(np.mean(reward))
    return {k: float(v) for k, v in m.items()}


def aggregate(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Ortalama, ornek std (ddof=1) ve ortalamanin bootstrap %95 araligi. NaN'ler dislanir."""
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    low, high = np.percentile(boot_means, _CI_PERCENT)
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return {"mean": float(arr.mean()), "std": std, "ci_low": float(low), "ci_high": float(high)}


def paired_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Eslestirilmis Wilcoxon isaretli sira testi (a - b). Tum farklar 0 ise p_value=1.0."""
    a_arr = np.asarray([np.nan if v is None else v for v in a], dtype=float)
    b_arr = np.asarray([np.nan if v is None else v for v in b], dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"a ve b ayni uzunlukta olmali: {a_arr.shape} != {b_arr.shape}")
    diff = a_arr - b_arr
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return {"mean_diff": math.nan, "median_diff": math.nan, "p_value": math.nan}
    if np.all(diff == 0.0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(diff).pvalue)
    return {"mean_diff": float(np.mean(diff)), "median_diff": float(np.median(diff)),
            "p_value": p_value}
