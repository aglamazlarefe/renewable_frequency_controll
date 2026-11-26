"""Mikro şebeke yük-frekans kontrolü Gymnasium ortamı (SPEC §3.4, §4, §6)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import BessParams, GridParams, ObsScales, RewardWeights, Scenario
from envs.dynamics import build_discrete_model

# Gözlem uzayının normalizasyon sınırı (SPEC §4: bileşenler [-1, 1]'e kırpılır)
_OBS_LIMIT = 1.0
# SOC sapması SPEC §4/§6'da 0.5'e bölünür (SOC ∈ [0,1] aralığının yarı genişliği)
_SOC_HALF_RANGE = 0.5
_SECONDS_PER_HOUR = 3600.0


class MicrogridLFCEnv(gym.Env):
    """BESS güç komutuyla frekans düzenleme ortamı."""

    metadata = {"render_modes": []}

    def __init__(self, scenario_generator=None, grid: GridParams = GridParams(),
                 bess: BessParams = BessParams(), reward: RewardWeights = RewardWeights(),
                 obs_scales: ObsScales = ObsScales()):
        super().__init__()
        self.scenario_generator = scenario_generator
        self.grid = grid
        self.bess = bess
        self.reward_weights = reward
        self.obs_scales = obs_scales
        self.dt = grid.dt

        self._ad, self._bd = build_discrete_model(grid, bess)

        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-_OBS_LIMIT, _OBS_LIMIT, (5,), dtype=np.float32)

        self.scenario: Scenario | None = None
        self.n_steps = 0
        self.step_count = 0
        self.delta_f_hz = 0.0
        self.rocof_hz_s = 0.0
        self.p_ess = 0.0
        self.u_ess = 0.0
        self.soc = bess.soc_ref
        self._x = np.zeros(5)
        self._integral = 0.0
        self._d_net = np.zeros(0)
        self._sum_f2 = 0.0
        self._throughput = 0.0

    # ── yardımcılar ────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        s = self.obs_scales
        raw = np.array([
            self.delta_f_hz / s.f_hz,
            self.rocof_hz_s / s.rocof_hz_s,
            self._integral / s.int_hz_s,
            self.p_ess / self.bess.p_max,
            (self.soc - self.bess.soc_ref) / _SOC_HALF_RANGE,
        ])
        return np.clip(raw, -_OBS_LIMIT, _OBS_LIMIT).astype(np.float32)

    def _info(self, d_net: float) -> dict:
        return {
            "delta_f_hz": float(self.delta_f_hz),
            "rocof_hz_s": float(self.rocof_hz_s),
            "p_ess": float(self.p_ess),
            "u_ess": float(self.u_ess),
            "soc": float(self.soc),
            "p_diesel": float(self._x[2]),
            "d_net": float(d_net),
            "shield_active": False,
        }

    # ── Gymnasium arayüzü ──────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = options.get("scenario") if options else None
        if scenario is None:
            if self.scenario_generator is None:
                from envs.scenarios import default_generator  # tembel içe aktarma
                self.scenario_generator = default_generator("train")
            scenario = self.scenario_generator.sample(self.np_random)
        self.scenario = scenario
        self.n_steps = scenario.n_steps
        self._d_net = np.asarray(scenario.d_net, dtype=float)
        self.step_count = 0
        self._x = np.zeros(5)
        self.delta_f_hz = 0.0
        self.rocof_hz_s = 0.0
        self.p_ess = 0.0
        self.u_ess = 0.0
        self.soc = float(scenario.soc0)
        self._integral = 0.0
        self._sum_f2 = 0.0
        self._throughput = 0.0
        return self._obs(), self._info(0.0)

    def step(self, action):
        bess, grid = self.bess, self.grid
        a = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        u_prev = self.u_ess
        u_target = a * bess.p_max
        max_delta = bess.ramp_rate * self.dt
        u = u_prev + float(np.clip(u_target - u_prev, -max_delta, max_delta))
        if self.soc <= bess.soc_min:
            u = min(u, 0.0)
        if self.soc >= bess.soc_max:
            u = max(u, 0.0)

        d_k = float(self._d_net[self.step_count])
        self._x = self._ad @ self._x + self._bd @ np.array([u, d_k])

        prev_f = self.delta_f_hz
        self.delta_f_hz = float(self._x[0] * grid.f0_hz)
        self.rocof_hz_s = (self.delta_f_hz - prev_f) / self.dt
        self.p_ess = float(self._x[4])
        self.u_ess = float(u)
        lim = self.obs_scales.int_hz_s
        self._integral = float(np.clip(self._integral + self.delta_f_hz * self.dt, -lim, lim))

        # SOC güncellemesi: adım sonu P_ess, tek yön verim
        e_mwh = self.p_ess * grid.s_base_mw * self.dt / _SECONDS_PER_HOUR
        if e_mwh >= 0.0:
            self.soc -= e_mwh / (bess.eta * bess.e_mwh)
        else:
            self.soc -= e_mwh * bess.eta / bess.e_mwh
        self.soc = float(np.clip(self.soc, 0.0, 1.0))

        w = self.reward_weights
        ef = self.delta_f_hz / w.f_ref_hz
        eu = self.p_ess / bess.p_max
        ej = (self.u_ess - u_prev) / bess.p_max
        es = (self.soc - bess.soc_ref) / _SOC_HALF_RANGE
        reward = float(-(w.w_f * ef ** 2 + w.w_u * eu ** 2 + w.w_j * ej ** 2 + w.w_soc * es ** 2))

        self.step_count += 1
        self._sum_f2 += self.delta_f_hz ** 2
        self._throughput += abs(self.p_ess) * self.dt
        truncated = bool(self.step_count >= self.n_steps)

        info = self._info(d_k)
        if truncated:
            info["ep_rmse_hz"] = float(np.sqrt(self._sum_f2 / self.step_count))
            info["ep_throughput_pu_s"] = float(self._throughput)
        return self._obs(), reward, False, truncated, info
