"""Kabul testleri — veri yükleyici ve senaryo üretimi (SPEC §3.1, §3.3, §5).

BU DOSYA SÖZLEŞMEDİR: değiştirmeyin. Yanlış olduğunu düşünüyorsanız gerekçesiyle raporlayın.
"""

import numpy as np
import pytest

from config import DEFAULT_DATA_PATH, GridParams, Scenario, ScenarioParams
from envs.scenarios import ScenarioGenerator, default_generator, make_scenario_bank
from utils.data_loader import load_wind_levels, split_wind_levels

N_HOURS = 2226


# ── veri yükleyici ────────────────────────────────────────────────

def test_load_wind_levels_shape_and_range():
    levels = load_wind_levels(DEFAULT_DATA_PATH)
    assert isinstance(levels, np.ndarray)
    assert levels.shape == (N_HOURS,)
    assert levels.dtype == np.float64
    assert np.all(np.isfinite(levels))
    assert levels.min() >= 0.0
    assert levels.max() == pytest.approx(1.0)
    # Türkçe sayı biçimi doğru okunmuş olmalı: ilk satır 6875,37 / en büyük 11875,27
    assert levels[0] == pytest.approx(6875.37 / 11875.27, rel=1e-6)


def test_load_wind_levels_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wind_levels(tmp_path / "yok.csv")


def test_load_wind_levels_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Tarih;Saat;Toplam\n20.11.2025;00:00;1,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_wind_levels(bad)


def test_split_is_chronological():
    levels = load_wind_levels()
    train, test = split_wind_levels(levels, train_frac=0.7)
    assert len(train) == int(0.7 * N_HOURS)
    assert len(train) + len(test) == N_HOURS
    np.testing.assert_array_equal(np.concatenate([train, test]), levels)


# ── senaryo üretimi ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def levels():
    return load_wind_levels()


def test_sample_structure(levels):
    params, grid = ScenarioParams(), GridParams()
    gen = ScenarioGenerator(levels, params, grid)
    n = params.n_steps(grid.dt)
    for seed in range(20):
        sc = gen.sample(np.random.default_rng(seed))
        assert isinstance(sc, Scenario)
        assert sc.d_wind.shape == (n,) and sc.d_load.shape == (n,)
        assert sc.n_steps == n
        assert np.all(np.isfinite(sc.d_wind)) and np.all(np.isfinite(sc.d_load))
        assert sc.d_wind[0] == 0.0 and sc.d_load[0] == 0.0
        assert params.soc0_range[0] <= sc.soc0 <= params.soc0_range[1]
        assert 0.0 <= sc.wind_level <= 1.0
        assert np.abs(sc.d_wind).max() < params.wind_share

        steps = sc.meta["load_steps"]
        assert params.n_load_steps[0] <= len(steps) <= params.n_load_steps[1]
        idx = sorted(int(k) for k, _ in steps)
        lo = params.load_step_window_s[0] / grid.dt
        hi = params.load_step_window_s[1] / grid.dt
        assert all(lo - 1e-9 <= k <= hi + 1e-9 for k in idx)
        gaps = np.diff(idx)
        assert np.all(gaps >= params.load_step_min_gap_s / grid.dt - 1e-9)
        for _, size in steps:
            assert params.load_step_size[0] - 1e-12 <= abs(size) <= params.load_step_size[1] + 1e-12
        # son değer ≈ adımların toplamı (gürültü payıyla)
        total = sum(size for _, size in steps)
        assert abs(sc.d_load[-1] - total) < 6 * params.load_noise_std + 1e-9


def test_sample_is_deterministic_given_rng(levels):
    gen = ScenarioGenerator(levels)
    a = gen.sample(np.random.default_rng(7))
    b = gen.sample(np.random.default_rng(7))
    c = gen.sample(np.random.default_rng(8))
    np.testing.assert_array_equal(a.d_wind, b.d_wind)
    np.testing.assert_array_equal(a.d_load, b.d_load)
    assert a.soc0 == b.soc0
    assert not np.array_equal(a.d_load, c.d_load)


def test_sample_does_not_touch_global_rng(levels):
    gen = ScenarioGenerator(levels)
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)
    gen.sample(np.random.default_rng(0))
    np.testing.assert_array_equal(np.random.random(5), expected)


def test_turbulence_scale(levels):
    """Rampasız senaryolarda türbülans std'si beklenen mertebede olmalı."""
    params = ScenarioParams(ramp_prob=0.0)
    gen = ScenarioGenerator(levels, params)
    ratios = []
    for seed in range(40):
        sc = gen.sample(np.random.default_rng(seed))
        if sc.wind_level < 0.2:
            continue
        expected = params.wind_share * sc.wind_level * params.turb_intensity
        ratios.append(np.std(sc.d_wind[300:]) / expected)
    assert 0.6 < float(np.mean(ratios)) < 1.3


def test_scenario_bank_reproducible(levels):
    a = make_scenario_bank(levels, n=5, seed=11)
    b = make_scenario_bank(levels, n=5, seed=11)
    c = make_scenario_bank(levels, n=5, seed=12)
    assert len(a) == 5
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.d_wind, y.d_wind)
        np.testing.assert_array_equal(x.d_load, y.d_load)
    assert any(not np.array_equal(x.d_load, z.d_load) for x, z in zip(a, c))
    # bankadaki senaryolar birbirinden farklı olmalı
    assert not np.array_equal(a[0].d_load, a[1].d_load)


def test_default_generators_use_disjoint_splits(levels):
    train_gen = default_generator("train")
    test_gen = default_generator("test")
    n_train = int(ScenarioParams().train_frac * N_HOURS)
    np.testing.assert_array_equal(train_gen.wind_levels, levels[:n_train])
    np.testing.assert_array_equal(test_gen.wind_levels, levels[n_train:])
    with pytest.raises(ValueError):
        default_generator("validation")
