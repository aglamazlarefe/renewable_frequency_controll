"""Ek birim testleri — dinamik model, ortam ve senaryo üretimi.

Kabul testlerinin (tests/acceptance/) kapsamadığı özellikleri sınar. Yalnızca
kamusal arayüzler kullanılır; hepsi hızlı ve belirlenimcidir.
"""

import dataclasses

import numpy as np
import pytest

from config import BessParams, GridParams, ObsScales, RewardWeights, Scenario, ScenarioParams
from envs.dynamics import build_continuous_model, build_discrete_model
from envs.microgrid_lfc_env import MicrogridLFCEnv
from envs.scenarios import ScenarioGenerator, make_scenario_bank

GRID = GridParams()
BESS = BessParams()
ZERO = np.zeros(1, dtype=np.float32)


def calm_scenario(n=300, soc0=0.5):
    return Scenario(d_wind=np.zeros(n), d_load=np.zeros(n), soc0=soc0, meta={"load_steps": []})


def make_env():
    """Veri dosyasına bağlı olmayan, sentetik rüzgâr seviyeli ortam."""
    gen = ScenarioGenerator(np.linspace(0.0, 1.0, 50))
    return MicrogridLFCEnv(scenario_generator=gen)


def run_actions(env, scenario, actions, seed=0):
    """Verilen aksiyon dizisiyle bölümü çalıştırır; info ve gözlem yörüngelerini döndürür."""
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    keys = ("delta_f_hz", "p_ess", "u_ess", "soc", "d_net")
    rec = {k: [] for k in keys}
    rec["reward"], rec["obs"] = [], [obs]
    info = {}
    for a in actions:
        obs, reward, _, truncated, info = env.step(np.array([a], dtype=np.float32))
        for k in keys:
            rec[k].append(info[k])
        rec["reward"].append(reward)
        rec["obs"].append(obs)
        if truncated:
            break
    out = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    out["last_info"] = info
    return out


# ── dinamik model ─────────────────────────────────────────────────

def test_discrete_eigenvalues_match_exp_of_continuous():
    """ZOH ayrıklaştırma: eig(Ad) = exp(eig(A)·dt) (çoklukla birlikte)."""
    a, _ = build_continuous_model(GRID, BESS)
    ad, _ = build_discrete_model(GRID, BESS)
    expected = list(np.exp(np.linalg.eigvals(a) * GRID.dt))
    actual = list(np.linalg.eigvals(ad))
    assert len(actual) == len(expected)
    for lam in actual:
        dist = [abs(lam - e) for e in expected]
        j = int(np.argmin(dist))
        assert dist[j] < 1e-8, f"{lam} için eşleşen sürekli kutup yok"
        expected.pop(j)  # her özdeğer yalnızca bir kez eşleşir


def test_dc_gain_disturbance_to_frequency_without_agc():
    """AGC kapalıyken kalıcı Δf_pu = d/β: yük artışı (d<0) için −|d|/β."""
    grid = dataclasses.replace(GRID, K_i_agc=0.0)
    ad, bd = build_discrete_model(grid, BESS)
    # AGC kapalıyken ΔP_c durumu sabittir (=0); kalan 4 durumlu altsistemi çöz
    idx = [0, 1, 2, 4]
    ad_r, bd_r = ad[np.ix_(idx, idx)], bd[idx]
    assert ad[3, 3] == pytest.approx(1.0) and np.allclose(ad[3, [0, 1, 2, 4]], 0.0)  # ΔP_c sabit
    x_ss = np.linalg.solve(np.eye(4) - ad_r, bd_r[:, 1])  # d = +1 pu için
    assert x_ss[0] == pytest.approx(1.0 / grid.beta, rel=1e-9)
    # Yük adımı (d = −size) kalıcı frekansı −size/β yapar
    assert -0.05 * x_ss[0] == pytest.approx(-0.05 / grid.beta, rel=1e-9)


def test_dc_gain_disturbance_to_frequency_matches_simulation():
    """Aynı DC kazancı, ayrık modelin uzun benzetimiyle de elde edilir."""
    grid = dataclasses.replace(GRID, K_i_agc=0.0)
    ad, bd = build_discrete_model(grid, BESS)
    x = np.zeros(5)
    for _ in range(int(600.0 / grid.dt)):
        x = ad @ x + bd @ np.array([0.0, 1.0])
    assert x[0] == pytest.approx(1.0 / grid.beta, rel=1e-6)


def test_bess_has_unit_dc_gain_from_command_to_power():
    """Sabit komut u için kalıcı P_ess = u (hem AGC açık hem kapalı)."""
    for k_agc in (GRID.K_i_agc, 0.0):
        grid = dataclasses.replace(GRID, K_i_agc=k_agc)
        ad, bd = build_discrete_model(grid, BESS)
        x = np.zeros(5)
        for _ in range(int(120.0 / grid.dt)):
            x = ad @ x + bd @ np.array([0.1, 0.0])
        assert x[4] == pytest.approx(0.1, rel=1e-9)
    # Çözümsel: P_ess satırı yalnızca kendi durumuna bağlı, Ad[4,4] + Bd[4,0] = 1
    ad, bd = build_discrete_model(GRID, BESS)
    assert ad[4, 4] + bd[4, 0] == pytest.approx(1.0, abs=1e-12)
    assert bd[4, 1] == 0.0


# ── ortam ─────────────────────────────────────────────────────────

def test_soc_energy_conservation_over_charge_discharge_cycle():
    """Eşit şebeke enerjisiyle şarj+deşarj: net SOC kaybı = verim kaybı."""
    n_half, n_tail = 300, 100
    p_max = BESS.p_max
    # +1, −1 simetrik kare dalga, ardından P_ess sönümlenene dek sıfır komut
    # Komut hız sınırı (2 pu/s · 0,1 s = p_max) +p_max → −p_max geçişini iki adıma yayar:
    # ters fazın ilk adımı 0 olur, bu yüzden ters faz bir adım uzun tutulur (Σu = 0).
    actions = [1.0] * n_half + [-1.0] * (n_half + 1) + [0.0] * n_tail
    env = make_env()
    tr = run_actions(env, calm_scenario(len(actions), soc0=0.5), actions)
    p = tr["p_ess"]
    assert abs(p[-1]) < 1e-9, "kuyruk sönümlenmedi"
    assert tr["u_ess"].max() == pytest.approx(p_max) and tr["u_ess"].min() == pytest.approx(-p_max)

    k = GRID.s_base_mw * GRID.dt / 3600.0  # pu·adım → MWh
    e_dis = float(p[p > 0].sum() * k)
    e_chg = float(-p[p < 0].sum() * k)
    assert e_dis == pytest.approx(e_chg, rel=1e-6), "şebeke enerjisi dengesiz"

    soc_drop = 0.5 - tr["soc"][-1]
    expected = (e_dis / BESS.eta - e_chg * BESS.eta) / BESS.e_mwh
    assert soc_drop == pytest.approx(expected, rel=1e-6)
    # Net kayıp pozitif ve (1/η − η)·E/E_kap'a eşit
    assert soc_drop > 0.0
    assert soc_drop == pytest.approx(e_dis * (1.0 / BESS.eta - BESS.eta) / BESS.e_mwh, rel=1e-6)


def test_reward_is_zero_only_at_equilibrium():
    env = make_env()
    # Denge: sıfır aksiyon, bozucu yok → ödül tam 0
    tr = run_actions(env, calm_scenario(50), [0.0] * 50)
    assert np.all(tr["reward"] == 0.0)
    # Dengeden her sapma (aksiyon ya da bozucu) ödülü kesin negatif yapar
    for action in (1.0, -1.0, 0.05, -0.05):
        tr = run_actions(env, calm_scenario(20), [action] * 20)
        assert np.all(tr["reward"] < 0.0), f"aksiyon {action}: ödül sıfır olmamalı"
    d_load = np.zeros(20)
    d_load[1:] = 0.01
    sc = Scenario(d_wind=np.zeros(20), d_load=d_load, meta={"load_steps": []})
    tr = run_actions(env, sc, [0.0] * 20)
    assert tr["reward"][0] == 0.0 and np.all(tr["reward"][1:] < 0.0)


def test_reward_never_positive_with_defaults():
    rng = np.random.default_rng(3)
    env = make_env()
    sc = make_scenario_bank(np.linspace(0, 1, 50), 1, seed=1)[0]
    tr = run_actions(env, sc, rng.uniform(-1, 1, sc.n_steps))
    assert np.all(tr["reward"] <= 0.0)


def test_observation_clipped_when_frequency_exceeds_scale():
    """|Δf| > ObsScales.f_hz iken gözlem tam ±1'e kırpılır, uzaydan taşmaz."""
    scales = ObsScales()
    n = 120
    d_load = np.zeros(n)
    d_load[1:] = 0.3  # NoBESS için Δf ≈ −0,7 Hz
    sc = Scenario(d_wind=np.zeros(n), d_load=d_load, meta={"load_steps": []})
    env = make_env()
    tr = run_actions(env, sc, [0.0] * n)
    assert tr["delta_f_hz"].min() < -scales.f_hz
    obs = tr["obs"]
    assert np.all(np.abs(obs) <= 1.0)
    clipped = tr["delta_f_hz"] < -scales.f_hz
    assert np.all(obs[1:][clipped, 0] == -1.0)
    # İntegral gözlemi de ±int_hz_s'te sınırlanır
    assert obs[-1, 2] == -1.0
    assert env.observation_space.contains(obs[-1].astype(np.float32))
    # Eşiğin altındaki bir sapma kırpılmaz
    small = tr["delta_f_hz"][0] / scales.f_hz
    assert abs(small) < 1.0 and obs[1, 0] == pytest.approx(small, rel=1e-5)


def test_reset_with_same_scenario_twice_gives_identical_trajectories():
    sc = make_scenario_bank(np.linspace(0, 1, 50), 3, seed=7)[1]
    actions = np.sin(np.arange(sc.n_steps) / 7.0)
    env = make_env()
    first = run_actions(env, sc, actions, seed=1)
    # Araya başka bir senaryo girer; tohum farklı olsa da sonuç aynı olmalı
    run_actions(env, make_scenario_bank(np.linspace(0, 1, 50), 3, seed=7)[2], actions, seed=5)
    second = run_actions(env, sc, actions, seed=2)
    for key in ("delta_f_hz", "p_ess", "u_ess", "soc", "reward", "obs"):
        assert np.array_equal(first[key], second[key]), key


def test_single_step_scenario_episode():
    sc = Scenario(d_wind=np.zeros(1), d_load=np.zeros(1), soc0=0.4, meta={"load_steps": []})
    env = make_env()
    obs, info = env.reset(seed=0, options={"scenario": sc})
    assert env.n_steps == 1 and env.soc == pytest.approx(0.4)
    obs, reward, terminated, truncated, info = env.step(ZERO)
    assert truncated is True and terminated is False
    assert reward == 0.0
    assert info["ep_rmse_hz"] == 0.0 and info["ep_throughput_pu_s"] == 0.0
    assert env.observation_space.contains(obs)


# ── senaryolar ────────────────────────────────────────────────────

def test_wind_ramp_metadata_consistent_with_d_wind():
    """Türbülans kapalıyken d_wind tam olarak meta'daki rampadır."""
    params = dataclasses.replace(ScenarioParams(), ramp_prob=1.0, turb_intensity=0.0)
    gen = ScenarioGenerator(np.linspace(0.1, 1.0, 20), params)
    dt = GRID.dt
    rng = np.random.default_rng(11)
    for _ in range(25):
        sc = gen.sample(rng)
        assert sc.meta["wind_ramp"] is not None
        start, end, size = sc.meta["wind_ramp"]
        n = sc.n_steps
        assert 0 <= start < end < n
        dur = end - start
        assert round(params.ramp_duration_s[0] / dt) <= dur <= round(params.ramp_duration_s[1] / dt)
        lo, hi = (v * params.wind_share for v in params.ramp_size)
        assert lo - 1e-12 <= abs(size) <= hi + 1e-12
        k = np.arange(n)
        expected = size * np.clip((k - start) / dur, 0.0, 1.0)
        assert np.allclose(sc.d_wind, expected, atol=1e-12)
        assert sc.d_wind[0] == 0.0
        assert np.allclose(sc.d_wind[end:], size)


def test_no_ramp_means_no_ramp_metadata():
    params = dataclasses.replace(ScenarioParams(), ramp_prob=0.0, turb_intensity=0.0)
    gen = ScenarioGenerator(np.linspace(0.1, 1.0, 20), params)
    rng = np.random.default_rng(2)
    for _ in range(5):
        sc = gen.sample(rng)
        assert sc.meta["wind_ramp"] is None
        assert np.all(sc.d_wind == 0.0)


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_extreme_wind_levels_produce_finite_scenarios(level):
    gen = ScenarioGenerator(np.array([level]))
    rng = np.random.default_rng(5)
    for _ in range(20):
        sc = gen.sample(rng)
        assert np.all(np.isfinite(sc.d_wind)) and np.all(np.isfinite(sc.d_load))
        assert np.isfinite(sc.soc0)
        assert sc.wind_level == level and sc.meta["hour_index"] == 0
        if level == 0.0:
            # Seviye 0: türbülans yok, yalnızca (varsa) rampa kalır
            ramp = sc.meta["wind_ramp"]
            if ramp is None:
                assert np.all(sc.d_wind == 0.0)
    # Ortamda da NaN üretmez
    env = MicrogridLFCEnv(scenario_generator=gen)
    tr = run_actions(env, gen.sample(rng), [0.0] * 20)
    assert np.all(np.isfinite(tr["delta_f_hz"]))


def test_scenario_bank_shared_prefix_independent_of_bank_size():
    """SeedSequence(seed).spawn(n) çocukları konumla indekslenir: n değişse de önek aynıdır."""
    levels = np.linspace(0.0, 1.0, 50)
    small = make_scenario_bank(levels, 4, seed=123)
    large = make_scenario_bank(levels, 9, seed=123)
    for a, b in zip(small, large[:4]):
        assert np.array_equal(a.d_wind, b.d_wind)
        assert np.array_equal(a.d_load, b.d_load)
        assert a.soc0 == b.soc0 and a.meta == b.meta
    # Bağımsızlık: farklı konumlar farklı senaryo, farklı tohum farklı banka
    assert not np.array_equal(large[0].d_load, large[1].d_load)
    other = make_scenario_bank(levels, 4, seed=124)
    assert not np.array_equal(small[0].d_load, other[0].d_load)
