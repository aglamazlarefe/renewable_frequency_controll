"""Klasik referans kontrolcüler ve RL sarmalayıcı (SPEC §3.6).

Klasik kontrolcüler fiziksel ölçümü info'dan okur; obs'tan ölçek çevirmez.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...


def _to_action(value: float) -> np.ndarray:
    """Skaler değeri [-1, 1]'e kırpılmış (1,) float32 diziye çevirir."""
    return np.array([np.clip(value, -1.0, 1.0)], dtype=np.float32)


class NoBessController:
    name = "NoBESS"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return _to_action(0.0)


class DroopController:
    name = "Droop"

    def __init__(self, k: float):
        self.k = k

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return _to_action(-self.k * info["delta_f_hz"])


class PIController:
    """Anti-windup'lı PI: doyumda integral dondurulur."""
    name = "PI"

    def __init__(self, kp: float, ki: float, dt: float):
        self.kp, self.ki, self.dt = kp, ki, dt
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        e = -info["delta_f_hz"]
        raw = self.kp * e + self.ki * self.integral
        out = float(np.clip(raw, -1.0, 1.0))
        if raw == out:
            self.integral += e * self.dt
        return _to_action(out)


class PIDController:
    """PID: türev terimi ölçümden (−kd·rocof), setpoint türevi yok."""
    name = "PID"

    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        e = -info["delta_f_hz"]
        raw = self.kp * e + self.ki * self.integral - self.kd * info["rocof_hz_s"]
        out = float(np.clip(raw, -1.0, 1.0))
        if raw == out:
            self.integral += e * self.dt
        return _to_action(out)


class RLController:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return _to_action(float(np.asarray(action).reshape(-1)[0]))
