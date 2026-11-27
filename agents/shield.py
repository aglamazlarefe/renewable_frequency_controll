"""Histerezisli PI güvenlik kalkanı (SPEC §3.5).

Kalkan yalnızca aksiyonu değiştirir; hız/güç/SOC sınırları ortamda uygulanır.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from config import ShieldParams


class ShieldedEnv(gym.Wrapper):
    """Frekans sapması eşiği aşınca ajan aksiyonunu PI çıkışıyla değiştiren sarmalayıcı."""

    def __init__(self, env: gym.Env, params: ShieldParams = ShieldParams()):
        super().__init__(env)
        self.params = params
        self._reset_state()

    def _reset_state(self) -> None:
        self._active = False
        self._integral = 0.0
        self._active_steps = 0
        self._total_steps = 0
        self._passive_steps = 0

    @property
    def _hold_steps(self) -> int:
        """min_hold_s süresinin adım karşılığı."""
        return math.ceil(self.params.min_hold_s / self.env.unwrapped.dt - 1e-9)

    def reset(self, **kwargs):
        self._reset_state()
        obs, info = self.env.reset(**kwargs)
        info["shield_active"] = False
        return obs, info

    def _update_activation(self, delta_f_hz: float) -> bool:
        """Histerezis kararı; devreye girişte True döner (darbesiz geçiş için)."""
        p = self.params
        engaged_now = False
        if not self._active:
            if abs(delta_f_hz) > p.engage_hz:
                self._active = True
                self._active_steps = 0
                engaged_now = True
        elif abs(delta_f_hz) < p.release_hz and self._active_steps >= self._hold_steps:
            self._active = False
        return engaged_now

    def _pi_action(self, delta_f_hz: float, engaged_now: bool) -> float:
        p = self.params
        unwrapped = self.env.unwrapped
        dt = unwrapped.dt
        e = -delta_f_hz
        if engaged_now:
            # Darbesiz geçiş: PI çıkışı mevcut normalize komuta eşit olsun
            u_norm = unwrapped.u_ess / unwrapped.bess.p_max
            self._integral = (u_norm - p.kp * e) / p.ki
        raw = p.kp * e + p.ki * self._integral
        a = float(np.clip(raw, -1.0, 1.0))
        if raw == a:  # doyumda integral dondurulur
            self._integral += e * dt
        return a

    def step(self, action):
        delta_f_hz = float(self.env.unwrapped.delta_f_hz)
        engaged_now = self._update_activation(delta_f_hz)
        if self._active:
            a = self._pi_action(delta_f_hz, engaged_now)
            action = np.array([a], dtype=np.float32)
            self._active_steps += 1
        else:
            self._passive_steps += 1
        self._total_steps += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        active = bool(self._active)
        info["shield_active"] = active
        reward = float(reward)
        info["reward_task"] = reward  # cezasız görev ödülü (metrikler bunu kullanır)
        if active:
            reward -= self.params.penalty
        if terminated or truncated:
            info["ep_autonomy"] = float(self._passive_steps / max(self._total_steps, 1))
        return obs, reward, bool(terminated), bool(truncated), info
