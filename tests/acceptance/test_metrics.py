"""Kabul testleri — metrikler ve istatistik (SPEC §3.8, §7).

BU DOSYA SÖZLEŞMEDİR: değiştirmeyin. Yanlış olduğunu düşünüyorsanız gerekçesiyle raporlayın.
"""

import math

import numpy as np
import pytest

from config import BessParams, GridParams, make_step_scenario
from utils.metrics import aggregate, compute_metrics, paired_test

METRIC_KEYS = {
    "rmse_mhz", "mae_mhz", "max_dev_mhz", "nadir_mhz", "zenith_mhz",
    "band_50_pct", "band_100_pct", "band_150_pct",
    "rocof_max_hz_s", "rocof_mean_hz_s", "settling_time_s",
    "bess_throughput_kwh", "bess_efc", "soc_min", "soc_max", "soc_final",
    "diesel_travel_pu", "control_jitter", "autonomy_pct", "mean_cost",
}


def make_traj(n=1000, dt=0.1, **overrides):
    traj = {
        "t": (np.arange(n) + 1) * dt,
        "delta_f_hz": np.zeros(n),
        "p_ess": np.zeros(n),
        "u_ess": np.zeros(n),
        "soc": np.full(n, 0.5),
        "p_diesel": np.zeros(n),
        "d_net": np.zeros(n),
        "shield_active": np.zeros(n, dtype=bool),
        "reward": np.zeros(n),
    }
    traj.update(overrides)
    return traj


def test_keys_and_types():
    m = compute_metrics(make_traj(), make_step_scenario(1000, 100, 0.05))
    assert METRIC_KEYS <= set(m)
    assert all(isinstance(v, float) for v in m.values())


def test_constant_deviation():
    m = compute_metrics(make_traj(delta_f_hz=np.full(1000, 0.1)))
    assert m["rmse_mhz"] == pytest.approx(100.0)
    assert m["mae_mhz"] == pytest.approx(100.0)
    assert m["max_dev_mhz"] == pytest.approx(100.0)
    assert m["zenith_mhz"] == pytest.approx(100.0)
    assert m["nadir_mhz"] == pytest.approx(100.0)
    assert m["band_50_pct"] == pytest.approx(0.0)
    assert m["band_100_pct"] == pytest.approx(0.0)      # sınır dahil değil
    assert m["band_150_pct"] == pytest.approx(100.0)
    assert m["rocof_max_hz_s"] == pytest.approx(0.0)


def test_nadir_zenith_and_bands():
    f = np.zeros(1000)
    f[:250] = -0.12
    f[250:500] = 0.06
    m = compute_metrics(make_traj(delta_f_hz=f))
    assert m["nadir_mhz"] == pytest.approx(-120.0)
    assert m["zenith_mhz"] == pytest.approx(60.0)
    assert m["max_dev_mhz"] == pytest.approx(120.0)
    assert m["band_50_pct"] == pytest.approx(50.0)
    assert m["band_100_pct"] == pytest.approx(75.0)
    assert m["band_150_pct"] == pytest.approx(100.0)
    assert m["rocof_max_hz_s"] == pytest.approx(0.18 / 0.1)


def test_throughput_and_cycles():
    grid, bess = GridParams(), BessParams()
    n = 3600  # 360 s
    p = np.full(n, 0.1)
    p[n // 2:] = -0.1
    m = compute_metrics(make_traj(n=n, p_ess=p), grid=grid, bess=bess)
    # 0.1 pu · 10 MW · 0.1 h = 0.1 MWh = 100 kWh
    assert m["bess_throughput_kwh"] == pytest.approx(100.0)
    assert m["bess_efc"] == pytest.approx(100.0 / (2 * bess.e_mwh * 1000))


def test_jitter_diesel_soc_autonomy_cost():
    n = 1000
    u = np.zeros(n)
    u[100:] = 0.1
    u[500:] = -0.1
    shield = np.zeros(n, dtype=bool)
    shield[:250] = True
    soc = np.linspace(0.6, 0.4, n)
    dies = np.zeros(n)
    dies[300:] = 0.05
    m = compute_metrics(make_traj(u_ess=u, shield_active=shield, soc=soc, p_diesel=dies,
                                  reward=np.full(n, -0.3)))
    assert m["control_jitter"] == pytest.approx((0.1 + 0.2) / BessParams().p_max)
    assert m["autonomy_pct"] == pytest.approx(75.0)
    assert m["soc_min"] == pytest.approx(0.4)
    assert m["soc_max"] == pytest.approx(0.6)
    assert m["soc_final"] == pytest.approx(0.4)
    assert m["diesel_travel_pu"] == pytest.approx(0.05)
    assert m["mean_cost"] == pytest.approx(0.3)


def test_settling_time():
    sc = make_step_scenario(1000, 100, 0.05)
    f = np.zeros(1000)
    f[100:150] = -0.1          # son bant dışı örnek k=149 → (149−100)·0.1 + 0.1 = 5.0 s
    m = compute_metrics(make_traj(delta_f_hz=f), sc, settle_band_hz=0.02)
    assert m["settling_time_s"] == pytest.approx(5.0)

    m = compute_metrics(make_traj(), sc)                       # bant hiç aşılmadı
    assert m["settling_time_s"] == pytest.approx(0.0)

    f = np.zeros(1000)
    f[100:] = -0.1                                              # hiç yerleşmedi
    m = compute_metrics(make_traj(delta_f_hz=f), sc)
    assert math.isnan(m["settling_time_s"])

    assert math.isnan(compute_metrics(make_traj(), None)["settling_time_s"])   # senaryo yok


def test_settling_window_ends_at_next_step():
    from config import Scenario
    d_load = np.zeros(1000)
    d_load[100:] = 0.05
    d_load[400:] = 0.10
    sc = Scenario(d_wind=np.zeros(1000), d_load=d_load, meta={"load_steps": [(100, 0.05), (400, 0.05)]})
    f = np.zeros(1000)
    f[100:130] = -0.1          # ilk adımın yanıtı: 3.0 s
    f[400:700] = -0.1          # ikinci adım — ilk adımın penceresi dışında
    m = compute_metrics(make_traj(delta_f_hz=f), sc)
    assert m["settling_time_s"] == pytest.approx(3.0)


def test_aggregate():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = aggregate(vals, n_boot=500, seed=0)
    assert set(a) == {"mean", "std", "ci_low", "ci_high"}
    assert a["mean"] == pytest.approx(3.0)
    assert a["std"] == pytest.approx(np.std(vals, ddof=1))
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["ci_low"] < a["ci_high"]
    assert aggregate(vals, n_boot=500, seed=0) == a           # deterministik
    # NaN'ler dışlanır
    b = aggregate([1.0, float("nan"), 3.0], n_boot=200, seed=0)
    assert b["mean"] == pytest.approx(2.0)
    # tümü NaN → NaN, hata değil
    c = aggregate([float("nan")], n_boot=50, seed=0)
    assert math.isnan(c["mean"])


def test_paired_test():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1, 30)
    better = base - 1.0 + rng.normal(0, 0.1, 30)
    r = paired_test(better, base)
    assert set(r) == {"mean_diff", "median_diff", "p_value"}
    assert r["mean_diff"] == pytest.approx(-1.0, abs=0.1)
    assert r["p_value"] < 0.001

    same = paired_test(base, base)
    assert same["p_value"] == pytest.approx(1.0)
    assert same["mean_diff"] == pytest.approx(0.0)

    noise = paired_test(base + rng.normal(0, 1, 30), base + rng.normal(0, 1, 30))
    assert 0.0 <= noise["p_value"] <= 1.0
