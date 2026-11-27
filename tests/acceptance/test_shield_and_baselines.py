"""Kabul testleri — güvenlik kalkanı, referans kontrolcüler, rollout (SPEC §3.5–3.7).

BU DOSYA SÖZLEŞMEDİR: değiştirmeyin. Yanlış olduğunu düşünüyorsanız gerekçesiyle raporlayın.
"""

import math

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from agents.baselines import (
    DroopController,
    NoBessController,
    PIController,
    PIDController,
    RLController,
)
from agents.shield import ShieldedEnv
from config import BaselineGains, GridParams, Scenario, ShieldParams, make_step_scenario
from envs.microgrid_lfc_env import MicrogridLFCEnv
from utils.rollout import run_episode

TRAJ_KEYS = {"t", "delta_f_hz", "p_ess", "u_ess", "soc", "p_diesel", "d_net", "shield_active", "reward"}
ZERO = np.zeros(1, dtype=np.float32)


def calm_scenario(n=300):
    return Scenario(d_wind=np.zeros(n), d_load=np.zeros(n), meta={"load_steps": []})


def run_zero_agent(env, scenario):
    """Sıfır aksiyonlu ajanla çalıştır; adım öncesi Δf ve kalkan bayrağını kaydet."""
    env.reset(seed=0, options={"scenario": scenario})
    pre, active, f, u = [], [], [], []
    truncated = False
    info = {}
    while not truncated:
        pre.append(env.unwrapped.delta_f_hz)
        _, _, _, truncated, info = env.step(ZERO)
        active.append(bool(info["shield_active"]))
        f.append(info["delta_f_hz"])
        u.append(info["u_ess"])
    return np.array(pre), np.array(active), np.array(f), np.array(u), info


# ── kalkan ────────────────────────────────────────────────────────

def test_shielded_env_passes_checker():
    check_env(ShieldedEnv(MicrogridLFCEnv()), warn=True)


def test_shield_inactive_when_calm():
    env = ShieldedEnv(MicrogridLFCEnv())
    _, active, f, _, info = run_zero_agent(env, calm_scenario())
    assert not active.any()
    assert info["ep_autonomy"] == pytest.approx(1.0)
    assert np.abs(f).max() < 1e-9


def test_shield_inactive_for_small_step():
    """Eşiğin altında kalan sapmada kalkan devreye girmez."""
    env = ShieldedEnv(MicrogridLFCEnv())
    _, active, f, _, _ = run_zero_agent(env, make_step_scenario(600, 100, 0.02))
    assert np.abs(f).max() < ShieldParams().engage_hz
    assert not active.any()


def test_shield_hysteresis_and_hold():
    params = ShieldParams()
    env = ShieldedEnv(MicrogridLFCEnv(), params)
    pre, active, f, _, info = run_zero_agent(env, make_step_scenario(1500, 100, 0.10))
    assert active.any()
    hold_steps = math.ceil(params.min_hold_s / env.unwrapped.dt - 1e-9)

    prev = False
    run_len = 0
    for k, a in enumerate(active):
        if a and not prev:
            assert abs(pre[k]) > params.engage_hz, f"k={k}: eşik aşılmadan devreye girdi"
            run_len = 0
        if prev and not a:
            assert abs(pre[k]) < params.release_hz, f"k={k}: bırakma eşiğinin üstünde bıraktı"
            assert run_len >= hold_steps, f"k={k}: min_hold_s dolmadan bıraktı"
        # pasifken eşik aşıldıysa mutlaka devreye girmeli
        if not a:
            assert abs(pre[k]) <= params.engage_hz or prev, f"k={k}: eşik aşıldı ama kalkan pasif"
        if a:
            run_len += 1
        prev = a

    assert info["ep_autonomy"] == pytest.approx(1.0 - active.mean())
    assert 0.0 < info["ep_autonomy"] < 1.0


def test_shield_improves_worst_case():
    sc = make_step_scenario(1500, 100, 0.10)
    _, _, f_raw, _, _ = run_zero_agent(MicrogridLFCEnv(), sc)
    _, _, f_sh, _, _ = run_zero_agent(ShieldedEnv(MicrogridLFCEnv()), sc)
    assert np.abs(f_sh).max() < np.abs(f_raw).max() - 0.03
    assert abs(f_sh[-1]) < 0.01


def test_shield_respects_actuator_limits():
    env = ShieldedEnv(MicrogridLFCEnv())
    bess, dt = env.unwrapped.bess, env.unwrapped.dt
    _, active, _, u, _ = run_zero_agent(env, make_step_scenario(1500, 100, 0.15))
    assert active.any()
    assert np.abs(u).max() <= bess.p_max + 1e-9
    assert np.abs(np.diff(np.concatenate([[0.0], u]))).max() <= bess.ramp_rate * dt + 1e-9
    # kalkan yük artışına deşarjla karşılık verir
    assert u[active].mean() > 0.0


def test_shield_penalty_applied():
    params = ShieldParams()
    sc = make_step_scenario(400, 50, 0.10)
    env = ShieldedEnv(MicrogridLFCEnv(), params)
    env.reset(seed=0, options={"scenario": sc})
    from config import RewardWeights
    w = RewardWeights()
    bess = env.unwrapped.bess
    u_prev = 0.0
    saw_active = False
    for _ in range(400):
        _, r, _, truncated, info = env.step(ZERO)
        base = -(w.w_f * (info["delta_f_hz"] / w.f_ref_hz) ** 2
                 + w.w_u * (info["p_ess"] / bess.p_max) ** 2
                 + w.w_j * ((info["u_ess"] - u_prev) / bess.p_max) ** 2
                 + w.w_soc * ((info["soc"] - bess.soc_ref) / 0.5) ** 2)
        expected = base - (params.penalty if info["shield_active"] else 0.0)
        assert r == pytest.approx(expected, rel=1e-6, abs=1e-9)
        saw_active |= bool(info["shield_active"])
        u_prev = info["u_ess"]
        if truncated:
            break
    assert saw_active


def test_shield_reset_clears_state():
    env = ShieldedEnv(MicrogridLFCEnv())
    # kalkan aktifken biten kısa bölüm
    run_zero_agent(env, make_step_scenario(130, 100, 0.15))
    _, active, _, _, info = run_zero_agent(env, calm_scenario())
    assert not active.any()
    assert info["ep_autonomy"] == pytest.approx(1.0)


# ── referans kontrolcüler ─────────────────────────────────────────

def controllers():
    g, dt = BaselineGains(), GridParams().dt
    return [
        NoBessController(),
        DroopController(g.droop_k),
        PIController(g.pi_kp, g.pi_ki, dt),
        PIDController(g.pid_kp, g.pid_ki, g.pid_kd, dt),
    ]


def test_controller_interface():
    env = MicrogridLFCEnv()
    obs, info = env.reset(seed=0)
    names = set()
    for c in controllers():
        c.reset()
        a = c.act(obs, info)
        assert isinstance(a, np.ndarray) and a.shape == (1,) and a.dtype == np.float32
        assert -1.0 <= float(a[0]) <= 1.0
        assert isinstance(c.name, str) and c.name
        names.add(c.name)
    assert names == {"NoBESS", "Droop", "PI", "PID"}


def test_controller_output_saturates():
    info = {"delta_f_hz": -5.0, "rocof_hz_s": -10.0}
    obs = np.zeros(5, dtype=np.float32)
    for c in controllers()[1:]:
        c.reset()
        assert float(c.act(obs, info)[0]) == pytest.approx(1.0)
    info = {"delta_f_hz": 5.0, "rocof_hz_s": 10.0}
    for c in controllers()[1:]:
        c.reset()
        assert float(c.act(obs, info)[0]) == pytest.approx(-1.0)


def test_pi_reset_clears_integral():
    g = BaselineGains()
    c = PIController(g.pi_kp, g.pi_ki, 0.1)
    obs = np.zeros(5, dtype=np.float32)
    for _ in range(20):
        c.act(obs, {"delta_f_hz": -0.02, "rocof_hz_s": 0.0})
    assert float(c.act(obs, {"delta_f_hz": 0.0, "rocof_hz_s": 0.0})[0]) > 0.0   # integral birikmiş
    c.reset()
    assert float(c.act(obs, {"delta_f_hz": 0.0, "rocof_hz_s": 0.0})[0]) == pytest.approx(0.0)


def test_run_episode_format():
    env = MicrogridLFCEnv()
    sc = make_step_scenario(400, 50, 0.05)
    traj = run_episode(env, NoBessController(), sc, seed=0)
    assert TRAJ_KEYS <= set(traj)
    for k in TRAJ_KEYS:
        assert len(traj[k]) == 400, k
    np.testing.assert_allclose(traj["t"][:3], [0.1, 0.2, 0.3])
    assert traj["shield_active"].dtype == bool
    assert np.abs(traj["p_ess"]).max() < 1e-12
    np.testing.assert_allclose(traj["d_net"], sc.d_net)


def test_run_episode_is_reproducible():
    sc = make_step_scenario(400, 50, 0.05)
    g = BaselineGains()
    a = run_episode(MicrogridLFCEnv(), PIController(g.pi_kp, g.pi_ki, 0.1), sc, seed=0)
    b = run_episode(MicrogridLFCEnv(), PIController(g.pi_kp, g.pi_ki, 0.1), sc, seed=0)
    for k in TRAJ_KEYS:
        np.testing.assert_array_equal(a[k], b[k])


def test_baselines_beat_no_bess_and_settle():
    """Varsayılan kazançlarla her BESS kontrolcüsü nadiri iyileştirir ve sistem sıfıra döner."""
    sc = make_step_scenario(1500, 100, 0.05)
    env = MicrogridLFCEnv()
    results = {c.name: run_episode(env, c, sc, seed=0) for c in controllers()}
    nadir_none = results["NoBESS"]["delta_f_hz"].min()
    assert nadir_none == pytest.approx(-0.233, abs=0.02)
    for name in ("Droop", "PI", "PID"):
        tr = results[name]
        assert tr["delta_f_hz"].min() > 0.6 * nadir_none, name       # nadir en az %40 daha sığ
        assert np.abs(tr["delta_f_hz"][-200:]).max() < 0.005, name    # frekans geri döndü
        assert np.abs(tr["p_ess"][-200:]).max() < 0.01, name          # batarya boşaltıldı (AGC)
        assert tr["p_ess"].max() > 0.02, name                         # batarya gerçekten çalıştı


def test_baselines_stable_for_large_step():
    sc = make_step_scenario(1500, 100, 0.10)
    env = MicrogridLFCEnv()
    for c in controllers():
        tr = run_episode(env, c, sc, seed=0)
        assert np.abs(tr["delta_f_hz"][-200:]).max() < 0.005, c.name
        assert np.abs(tr["delta_f_hz"]).max() < 0.6, c.name


class _FakeModel:
    def predict(self, obs, deterministic=True):
        assert deterministic is True
        return np.array([0.25], dtype=np.float32), None


def test_rl_controller_wraps_model():
    c = RLController(_FakeModel(), name="TD3+Shield")
    assert c.name == "TD3+Shield"
    c.reset()
    a = c.act(np.zeros(5, dtype=np.float32), {})
    assert a.shape == (1,) and a.dtype == np.float32
    assert float(a[0]) == pytest.approx(0.25)
