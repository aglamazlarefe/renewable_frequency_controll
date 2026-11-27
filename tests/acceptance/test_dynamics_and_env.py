"""Kabul testleri — dinamik model ve Gymnasium ortamı (SPEC §2, §3.2, §3.4, §4, §6).

BU DOSYA SÖZLEŞMEDİR: değiştirmeyin. Yanlış olduğunu düşünüyorsanız gerekçesiyle raporlayın.
"""

import dataclasses

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from config import BessParams, GridParams, Scenario, make_step_scenario
from envs.dynamics import STATE_NAMES, build_continuous_model, build_discrete_model
from envs.microgrid_lfc_env import MicrogridLFCEnv

INFO_KEYS = {"delta_f_hz", "rocof_hz_s", "p_ess", "u_ess", "soc", "p_diesel", "d_net", "shield_active"}


def calm_scenario(n=300, soc0=0.5):
    return Scenario(d_wind=np.zeros(n), d_load=np.zeros(n), soc0=soc0, meta={"load_steps": []})


def run_const(env, scenario, action, seed=0):
    """Sabit aksiyonla bir bölüm çalıştır; info yörüngelerini döndür."""
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    out = {k: [] for k in ("delta_f_hz", "p_ess", "u_ess", "soc", "d_net", "reward")}
    obs_list = [obs]
    truncated = terminated = False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        steps += 1
        obs_list.append(obs)
        for k in ("delta_f_hz", "p_ess", "u_ess", "soc", "d_net"):
            out[k].append(info[k])
        out["reward"].append(reward)
        assert steps <= scenario.n_steps, "bölüm n_steps'te kesilmedi"
    out = {k: np.array(v, dtype=float) for k, v in out.items()}
    out["obs"] = np.array(obs_list)
    out["terminated"] = terminated
    out["steps"] = steps
    return out


# ── dinamik model ─────────────────────────────────────────────────

def test_state_order():
    assert STATE_NAMES == ("delta_f_pu", "dp_g", "dp_m", "dp_c", "p_ess")


def test_model_shapes():
    A, B = build_continuous_model(GridParams(), BessParams())
    Ad, Bd = build_discrete_model(GridParams(), BessParams())
    assert A.shape == (5, 5) and B.shape == (5, 2)
    assert Ad.shape == (5, 5) and Bd.shape == (5, 2)


def test_open_loop_is_stable():
    Ad, _ = build_discrete_model(GridParams(), BessParams())
    assert np.abs(np.linalg.eigvals(Ad)).max() < 1.0


def test_zoh_matches_fine_integration():
    """ZOH ayrıklaştırma, çok küçük adımlı Euler ile aynı sonucu vermeli."""
    grid, bess = GridParams(), BessParams()
    A, B = build_continuous_model(grid, bess)
    Ad, Bd = build_discrete_model(grid, bess)
    v = np.array([0.1, -0.05])
    x_fine = np.zeros(5)
    h = 1e-4
    for _ in range(int(round(1.0 / h))):
        x_fine = x_fine + h * (A @ x_fine + B @ v)
    x_zoh = np.zeros(5)
    for _ in range(int(round(1.0 / grid.dt))):
        x_zoh = Ad @ x_zoh + Bd @ v
    np.testing.assert_allclose(x_zoh, x_fine, atol=2e-4)


def test_primary_only_steady_state():
    """AGC kapalıyken kalıcı durum: Δf = −ΔP / (D + 1/R)."""
    grid = dataclasses.replace(GridParams(), K_i_agc=0.0)
    Ad, Bd = build_discrete_model(grid, BessParams())
    step = 0.05
    x = np.zeros(5)
    for _ in range(int(60 / grid.dt)):
        x = Ad @ x + Bd @ np.array([0.0, -step])
    expected_pu = -step / grid.beta
    assert x[0] == pytest.approx(expected_pu, rel=0.02)
    assert x[0] * grid.f0_hz == pytest.approx(-0.119, abs=0.004)


def test_agc_restores_frequency():
    grid = GridParams()
    Ad, Bd = build_discrete_model(grid, BessParams())
    x = np.zeros(5)
    nadir = 0.0
    for _ in range(int(150 / grid.dt)):
        x = Ad @ x + Bd @ np.array([0.0, -0.05])
        nadir = min(nadir, x[0] * grid.f0_hz)
    assert abs(x[0] * grid.f0_hz) < 0.005          # frekans geri döndü
    assert x[2] == pytest.approx(0.05, abs=0.002)  # dizel yükü devraldı
    assert -0.40 < nadir < -0.12                    # nadir makul aralıkta


def test_positive_load_step_lowers_frequency_and_bess_raises_it():
    grid = GridParams()
    Ad, Bd = build_discrete_model(grid, BessParams())
    x_load = Ad @ np.zeros(5) + Bd @ np.array([0.0, -0.05])
    assert x_load[0] < 0
    x = np.zeros(5)
    for _ in range(10):
        x = Ad @ x + Bd @ np.array([0.1, 0.0])
    assert x[0] > 0 and x[4] > 0


# ── ortam ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    e = MicrogridLFCEnv()
    yield e
    e.close()


def test_env_passes_sb3_checker():
    check_env(MicrogridLFCEnv(), warn=True)


def test_spaces(env):
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (5,)
    assert env.observation_space.dtype == np.float32
    assert float(env.observation_space.low.min()) == -1.0
    assert float(env.observation_space.high.max()) == 1.0


def test_reset_and_step_info_keys(env):
    obs, info = env.reset(seed=1)
    assert INFO_KEYS <= set(info)
    assert obs.shape == (5,) and obs.dtype == np.float32
    assert info["delta_f_hz"] == 0.0 and info["shield_active"] is False
    obs, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert INFO_KEYS <= set(info)
    assert isinstance(reward, float)
    assert terminated is False


def test_truncation_not_termination(env):
    sc = calm_scenario(n=50)
    out = run_const(env, sc, 0.0)
    assert out["steps"] == 50
    assert out["terminated"] is False
    assert env.n_steps == 50


def test_episode_end_info(env):
    sc = make_step_scenario(200, 20, 0.05)
    obs, info = env.reset(seed=0, options={"scenario": sc})
    truncated = False
    while not truncated:
        obs, r, term, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert info["ep_rmse_hz"] > 0.0
    assert info["ep_throughput_pu_s"] == pytest.approx(0.0, abs=1e-9)


def test_equilibrium_stays_at_rest(env):
    out = run_const(env, calm_scenario(), 0.0)
    assert np.abs(out["delta_f_hz"]).max() < 1e-9
    assert np.abs(out["p_ess"]).max() < 1e-12
    assert np.ptp(out["soc"]) < 1e-12
    assert np.abs(out["reward"]).max() < 1e-12
    assert np.abs(out["obs"][:, :4]).max() < 1e-9


def test_scenario_injection(env):
    rng = np.random.default_rng(3)
    n = 120
    sc = Scenario(d_wind=rng.normal(0, 0.01, n), d_load=rng.normal(0, 0.01, n), meta={"load_steps": []})
    out = run_const(env, sc, 0.0)
    np.testing.assert_allclose(out["d_net"], sc.d_wind - sc.d_load, atol=1e-12)
    assert env.scenario is sc


def test_determinism_and_seed_sensitivity():
    def trajectory(seed):
        e = MicrogridLFCEnv()
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(99)
        rows = [obs]
        for _ in range(200):
            obs, *_ = e.step(rng.uniform(-1, 1, size=1).astype(np.float32))
            rows.append(obs)
        return np.array(rows)

    a, b, c = trajectory(5), trajectory(5), trajectory(6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_env_ignores_global_numpy_seed():
    def trajectory(global_seed):
        np.random.seed(global_seed)
        e = MicrogridLFCEnv()
        obs, _ = e.reset(seed=5)
        rows = [obs]
        for _ in range(100):
            obs, *_ = e.step(np.zeros(1, dtype=np.float32))
            rows.append(obs)
        return np.array(rows)

    np.testing.assert_array_equal(trajectory(1), trajectory(2))


def test_load_step_sign_and_reward(env):
    out = run_const(env, make_step_scenario(300, 50, 0.05), 0.0)
    assert np.abs(out["delta_f_hz"][:49]).max() < 1e-9
    assert out["delta_f_hz"][60] < 0.0
    assert out["delta_f_hz"].min() == pytest.approx(-0.233, abs=0.02)
    assert np.all(out["reward"] <= 1e-12)
    assert out["reward"].min() < -1.0   # 233 mHz, f_ref=100 mHz → ef² > 1


def test_actuator_limits(env):
    bess, dt = env.bess, env.dt
    out = run_const(env, calm_scenario(n=100), 1.0)
    u = np.concatenate([[0.0], out["u_ess"]])
    assert np.all(np.diff(u) <= bess.ramp_rate * dt + 1e-9)
    assert out["u_ess"].max() == pytest.approx(bess.p_max)
    assert np.all(np.abs(out["p_ess"]) <= bess.p_max + 1e-9)
    assert out["p_ess"][-1] == pytest.approx(bess.p_max, rel=1e-3)
    # deşarj frekansı yükseltir ve SOC'yi düşürür
    assert out["delta_f_hz"][5] > 0.0
    assert out["soc"][-1] < 0.5

    out_neg = run_const(env, calm_scenario(n=100), -1.0)
    assert out_neg["u_ess"].min() == pytest.approx(-bess.p_max)
    assert out_neg["soc"][-1] > 0.5


def test_ramp_limit_is_enforced():
    bess = dataclasses.replace(BessParams(), ramp_rate=0.4)
    e = MicrogridLFCEnv(bess=bess)
    out = run_const(e, calm_scenario(n=30), 1.0)
    u = np.concatenate([[0.0], out["u_ess"]])
    assert np.diff(u).max() == pytest.approx(0.4 * e.dt)
    assert out["u_ess"][0] == pytest.approx(0.04)
    assert out["u_ess"][3] == pytest.approx(0.16)
    assert out["u_ess"][10] == pytest.approx(bess.p_max)


def test_soc_efficiency(env):
    """Aynı |güç| için deşarj SOC'yi şarjın doldurduğundan daha hızlı boşaltır."""
    bess = env.bess
    dis = run_const(env, calm_scenario(n=200), 1.0)
    chg = run_const(env, calm_scenario(n=200), -1.0)
    d_dis = 0.5 - dis["soc"][-1]
    d_chg = chg["soc"][-1] - 0.5
    assert d_dis > 0 and d_chg > 0
    assert d_dis / d_chg == pytest.approx(1.0 / bess.eta**2, rel=0.01)
    # enerji muhasebesi: Σ P·dt (MWh) / (eta·E)
    e_mwh = dis["p_ess"].sum() * env.grid.s_base_mw * env.dt / 3600.0
    assert d_dis == pytest.approx(e_mwh / (bess.eta * bess.e_mwh), rel=1e-6)


def test_soc_lower_limit_blocks_discharge(env):
    bess = env.bess
    out = run_const(env, calm_scenario(n=200, soc0=bess.soc_min + 0.0005), 1.0)
    assert out["soc"].min() >= bess.soc_min - 1e-3
    assert abs(out["p_ess"][-1]) < 1e-3
    below = out["soc"][:-1] <= bess.soc_min
    assert below.any()
    assert np.all(out["u_ess"][1:][below] <= 1e-12)


def test_soc_upper_limit_blocks_charge(env):
    bess = env.bess
    out = run_const(env, calm_scenario(n=200, soc0=bess.soc_max - 0.0005), -1.0)
    assert out["soc"].max() <= bess.soc_max + 1e-3
    assert abs(out["p_ess"][-1]) < 1e-3


def test_observation_definition(env):
    """Gözlem bileşenleri SPEC §4'teki ölçeklerle üretilmeli."""
    from config import ObsScales
    sc = make_step_scenario(200, 20, 0.03)
    obs, info = env.reset(seed=0, options={"scenario": sc})
    scales = ObsScales()
    prev_f = 0.0
    integral = 0.0
    for k in range(150):
        obs, r, term, trunc, info = env.step(np.array([0.2], dtype=np.float32))
        f = info["delta_f_hz"]
        integral = float(np.clip(integral + f * env.dt, -scales.int_hz_s, scales.int_hz_s))
        expected = np.array([
            np.clip(f / scales.f_hz, -1, 1),
            np.clip((f - prev_f) / env.dt / scales.rocof_hz_s, -1, 1),
            np.clip(integral / scales.int_hz_s, -1, 1),
            np.clip(info["p_ess"] / env.bess.p_max, -1, 1),
            np.clip((info["soc"] - env.bess.soc_ref) / 0.5, -1, 1),
        ])
        np.testing.assert_allclose(obs, expected, atol=1e-5)
        assert info["rocof_hz_s"] == pytest.approx((f - prev_f) / env.dt, abs=1e-9)
        prev_f = f
    assert np.all(np.abs(obs) <= 1.0)


def test_reward_definition(env):
    from config import RewardWeights
    w = RewardWeights()
    sc = make_step_scenario(100, 10, 0.05)
    env.reset(seed=0, options={"scenario": sc})
    u_prev = 0.0
    rng = np.random.default_rng(0)
    for _ in range(80):
        a = rng.uniform(-1, 1, size=1).astype(np.float32)
        _, r, _, _, info = env.step(a)
        ef = info["delta_f_hz"] / w.f_ref_hz
        eu = info["p_ess"] / env.bess.p_max
        ej = (info["u_ess"] - u_prev) / env.bess.p_max
        es = (info["soc"] - env.bess.soc_ref) / 0.5
        expected = -(w.w_f * ef**2 + w.w_u * eu**2 + w.w_j * ej**2 + w.w_soc * es**2)
        assert r == pytest.approx(expected, rel=1e-6, abs=1e-9)
        u_prev = info["u_ess"]
