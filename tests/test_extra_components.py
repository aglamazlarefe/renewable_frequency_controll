"""Ek birim testleri — kalkan, referans kontrolcüler, metrikler ve rapor.

Kabul testlerinin kapsamadığı özellikleri sınar. Kalkanın iç mantığı, gerçek
ortam yerine bozucuyu doğrudan süren küçük bir sahte ortamla denenir (yalnızca
kamusal arayüz: ShieldedEnv.step/reset ve info anahtarları).
"""

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from agents.baselines import DroopController, PIController, PIDController
from agents.shield import ShieldedEnv
from config import (
    ALL_CONTROLLERS,
    BessParams,
    GridParams,
    Scenario,
    ShieldParams,
    make_step_scenario,
)
from envs.microgrid_lfc_env import MicrogridLFCEnv
from envs.scenarios import ScenarioGenerator
from utils.metrics import aggregate, paired_test
from utils.report import MISSING, render_results_table
from utils.rollout import run_episode

DT = GridParams().dt
P_MAX = BessParams().p_max


class FakeFreqEnv(gym.Env):
    """Δf'i testin doğrudan belirlediği sahte ortam; uygulanan aksiyonları kaydeder."""

    observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.dt = DT
        self.bess = BessParams()
        self.delta_f_hz = 0.0
        self.u_ess = 0.0
        self.applied = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.delta_f_hz, self.u_ess = 0.0, 0.0
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        self.applied.append(a)
        self.u_ess = a * self.bess.p_max
        return np.zeros(5, dtype=np.float32), 0.0, False, False, {}


def drive(shielded, df_sequence):
    """Her adımdan önce Δf'i ayarla, ajan aksiyonu 0; shield_active ve uygulanan aksiyonlar."""
    shielded.reset()
    fake = shielded.unwrapped
    active = []
    for df in df_sequence:
        fake.delta_f_hz = df
        _, _, _, _, info = shielded.step(np.zeros(1, dtype=np.float32))
        active.append(info["shield_active"])
    return np.array(active), np.array(fake.applied)


def true_runs(flags):
    """Ardışık True bloklarının uzunlukları ve (blok sonunda bırakıldı mı) bilgisi."""
    runs, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            runs.append((n, True))
            n = 0
    if n:
        runs.append((n, False))
    return runs


# ── kalkan ────────────────────────────────────────────────────────

def test_shield_bumpless_engage_matches_previous_command():
    n = 600
    d_load = np.zeros(n)
    d_load[50:] = 0.15
    sc = Scenario(d_wind=np.zeros(n), d_load=d_load, meta={"load_steps": [(50, 0.15)]})
    env = ShieldedEnv(MicrogridLFCEnv(scenario_generator=ScenarioGenerator(np.array([0.5]))))
    _, info = env.reset(seed=0, options={"scenario": sc})
    ramp_limit = BessParams().ramp_rate * DT
    u_prev = info["u_ess"]
    first = None
    for k in range(n):
        # Ajan bataryayı yanlış yöne (şarj) sürüyor: komut sıfırdan farklı
        _, _, _, truncated, info = env.step(np.array([-0.5], dtype=np.float32))
        if info["shield_active"]:
            first = k
            break
        u_prev = info["u_ess"]
    assert first is not None, "kalkan hiç devreye girmedi"
    assert abs(u_prev) > 1e-3, "test anlamlı değil: önceki komut sıfır"
    assert abs(info["u_ess"] - u_prev) <= ramp_limit + 1e-12
    assert info["u_ess"] == pytest.approx(u_prev, abs=1e-6)  # darbesiz: komut sıçramaz


def test_shield_integrator_freezes_in_saturation():
    """Uzun doyumdan sonra integral birikmemiş olmalı: hata küçülünce çıkış hemen düşer."""
    shielded = ShieldedEnv(FakeFreqEnv())
    # 1 Hz sapma: kp·e = 5 ≫ 1 → doyum; 100 adım boyunca
    seq = [-1.0] * 100 + [-0.15] * 3
    _, applied = drive(shielded, seq)
    assert np.all(applied[5:100] == 1.0), "doyumda çıkış +1'de kalmalı"
    # Birikmiş (windup) integral olsaydı çıkış hâlâ +1 olurdu; donmuşsa negatife iner
    assert applied[100] < 0.0


def test_shield_hold_time_respected_under_rapid_oscillation():
    params = ShieldParams()
    hold = math.ceil(params.min_hold_s / DT - 1e-9)
    seq = [0.3, 0.0] * 40  # her adım eşik üstü / bırakma altı
    active, _ = drive(ShieldedEnv(FakeFreqEnv(), params), seq)
    runs = true_runs(active)
    assert len(runs) >= 2
    for length, released in runs:
        if released:
            assert length >= hold, f"{length} adım aktif kaldı, en az {hold} gerekir"
    # Karşı deney: bekleme süresi sıfırsa kalkan her adım titreşir
    no_hold = ShieldParams(min_hold_s=0.0)
    active0, _ = drive(ShieldedEnv(FakeFreqEnv(), no_hold), seq)
    assert all(length == 1 for length, _ in true_runs(active0))


class _InfoSpy(gym.Wrapper):
    """Son adımın info sözlüğünü saklar."""

    def step(self, action):
        out = super().step(action)
        self.last_info = out[4]
        return out


class _RewardSpy(gym.Wrapper):
    """İçteki (kalkan cezasız) ödülü kaydeder."""

    def reset(self, **kwargs):
        self.inner_rewards = []
        return super().reset(**kwargs)

    def step(self, action):
        out = super().step(action)
        self.inner_rewards.append(out[1])
        return out


class _ZeroController:
    name = "zero"

    def reset(self):
        pass

    def act(self, obs, info):
        return np.zeros(1, dtype=np.float32)


def test_ep_autonomy_equals_one_minus_mean_shield_active():
    sc = make_step_scenario(1000, 100, 0.10)
    spy = _InfoSpy(ShieldedEnv(MicrogridLFCEnv(scenario_generator=ScenarioGenerator(np.array([0.5])))))
    traj = run_episode(spy, _ZeroController(), sc)
    assert traj["shield_active"].any()
    assert spy.last_info["ep_autonomy"] == pytest.approx(1.0 - traj["shield_active"].mean())


def test_reward_task_equals_unshielded_reward_and_penalty_is_extra():
    params = ShieldParams()
    inner = _RewardSpy(MicrogridLFCEnv(scenario_generator=ScenarioGenerator(np.array([0.5]))))
    env = ShieldedEnv(inner, params)
    env.reset(seed=0, options={"scenario": make_step_scenario(600, 50, 0.10)})
    n_active = 0
    for k in range(600):
        _, reward, _, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        assert info["reward_task"] == inner.inner_rewards[-1]
        expected = info["reward_task"] - (params.penalty if info["shield_active"] else 0.0)
        assert reward == pytest.approx(expected, abs=1e-12)
        n_active += info["shield_active"]
        if truncated:
            break
    assert n_active > 0, "kalkan hiç devreye girmedi: test anlamsız"


# ── referans kontrolcüler ────────────────────────────────────────

def _info(df, rocof=0.0):
    return {"delta_f_hz": df, "rocof_hz_s": rocof}


OBS = np.zeros(5, dtype=np.float32)


def test_pid_derivative_sign_rising_frequency_gives_negative_contribution():
    kd = 0.5
    pid = PIDController(kp=5.0, ki=2.0, kd=kd, dt=DT)
    # Δf = 0 (P ve I terimi yok); frekans yükseliyor → türev katkısı −kd·rocof < 0
    assert pid.act(OBS, _info(0.0, rocof=0.4))[0] == pytest.approx(-kd * 0.4)
    pid.reset()
    assert pid.act(OBS, _info(0.0, rocof=-0.4))[0] == pytest.approx(+kd * 0.4)
    # Aynı Δf'te PID çıkışı PI'dan, yükselen frekansta küçüktür
    pi = PIController(5.0, 2.0, DT)
    pid.reset()
    assert pid.act(OBS, _info(0.02, rocof=0.3))[0] < pi.act(OBS, _info(0.02, rocof=0.3))[0]


def test_droop_is_memoryless():
    k = 7.0
    d = DroopController(k)
    d.act(OBS, _info(-0.15))
    d.act(OBS, _info(0.09))
    after_history = d.act(OBS, _info(0.03))
    fresh = DroopController(k).act(OBS, _info(0.03))
    assert np.array_equal(after_history, fresh)
    assert after_history[0] == pytest.approx(-k * 0.03)
    assert after_history.shape == (1,) and after_history.dtype == np.float32


class _NaivePI:
    """Anti-windup'sız referans: integral doyumda da birikir."""

    def __init__(self, kp, ki, dt):
        self.kp, self.ki, self.dt, self.i = kp, ki, dt, 0.0

    def act(self, df):
        e = -df
        self.i += e * self.dt
        return float(np.clip(self.kp * e + self.ki * self.i, -1.0, 1.0))


def test_pi_anti_windup_recovers_faster_than_naive_integrator():
    kp, ki = 1.0, 1.0
    pi, naive = PIController(kp, ki, DT), _NaivePI(kp, ki, DT)
    # Faz 1: uzun süre doyumlu pozitif hata (Δf = −1 Hz)
    for _ in range(100):
        out_pi = pi.act(OBS, _info(-1.0))[0]
        out_naive = naive.act(-1.0)
    assert out_pi == 1.0 and out_naive == 1.0
    # Faz 2: hata işaret değiştirir (Δf = +0,5 Hz); çıkışın 0'ın altına inmesi için gereken adım
    def steps_to_negative(act):
        for n in range(1, 1000):
            if act(0.5) < 0.0:
                return n
        return 1000

    n_pi = steps_to_negative(lambda df: pi.act(OBS, _info(df))[0])
    n_naive = steps_to_negative(naive.act)
    assert n_pi <= 3
    assert n_naive > 50
    assert n_pi * 10 < n_naive


# ── metrikler ─────────────────────────────────────────────────────

def test_aggregate_ci_shrinks_with_more_samples():
    data = np.random.default_rng(0).normal(10.0, 2.0, size=400)
    small = aggregate(data[:10])
    large = aggregate(data[:400])
    assert (large["ci_high"] - large["ci_low"]) < 0.5 * (small["ci_high"] - small["ci_low"])
    for agg in (small, large):
        assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]
    # Ortalamanın CI'si ~ std/sqrt(n) mertebesinde (kaba doğrulama)
    assert (large["ci_high"] - large["ci_low"]) == pytest.approx(
        2 * 1.96 * large["std"] / math.sqrt(400), rel=0.25)


def test_paired_test_sign_convention_better_a_gives_negative_diff():
    rng = np.random.default_rng(1)
    b = rng.uniform(1.0, 2.0, size=20)          # maliyet (düşük = iyi)
    a = b - rng.uniform(0.1, 0.5, size=20)      # a her senaryoda daha iyi
    res = paired_test(a, b)
    assert res["mean_diff"] < 0.0 and res["median_diff"] < 0.0
    assert res["mean_diff"] == pytest.approx(float(np.mean(a - b)))
    assert res["p_value"] < 0.01
    flipped = paired_test(b, a)
    assert flipped["mean_diff"] == pytest.approx(-res["mean_diff"])
    assert flipped["p_value"] == pytest.approx(res["p_value"])


# ── rapor ─────────────────────────────────────────────────────────

def _entry(mean, std):
    return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean}


def test_render_table_with_none_values_and_single_controller():
    results = {"controllers": {"PI": {"metrics": {
        "rmse_mhz": _entry(12.34, 1.5),
        "max_dev_mhz": {"mean": None, "std": None, "ci_low": None, "ci_high": None},
        "settling_time_s": {"mean": 3.0, "std": None},
        # band_100_pct ve diğerleri hiç yok
    }}}}
    table = render_results_table(results)
    lines = table.strip().split("\n")
    assert lines[0] == "| Metrik | PI |"
    assert lines[1] == "|---|---|"
    assert len(lines) == 2 + 8
    rows = {line.split("|")[1].strip(): line.split("|")[2].strip() for line in lines[2:]}
    assert rows["RMSE [mHz]"] == "12.3 ± 1.5"
    assert rows["Maks. sapma [mHz]"] == MISSING
    assert rows["Yerleşme süresi [s]"] == "3.0"
    assert rows["Ortalama maliyet"] == MISSING
    assert render_results_table(results) == table  # deterministik


def test_render_table_rows_have_consistent_column_count():
    results = {"controllers": {"PI": {"metrics": {"band_100_pct": _entry(99.0, 1.0)}}}}
    lines = render_results_table(results).strip().split("\n")
    # Tek kontrolcü: her satırda tam 3 ayraç ('|') olmalı
    assert all(line.replace("\\|", "").count("|") == 3 for line in lines)


def test_render_table_orders_columns_by_all_controllers():
    metrics = {"rmse_mhz": _entry(1.0, 0.1)}
    shuffled = {"controllers": {name: {"metrics": metrics} for name in reversed(ALL_CONTROLLERS)}}
    header = render_results_table(shuffled).split("\n")[0]
    cols = [c.strip() for c in header.strip("|").split("|")][1:]
    assert cols == list(ALL_CONTROLLERS)
