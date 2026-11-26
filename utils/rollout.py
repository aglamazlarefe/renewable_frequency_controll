"""Tek bölüm çalıştırma (SPEC §3.7)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from agents.baselines import Controller
from config import Scenario


def run_episode(env: gym.Env, controller: Controller, scenario: Scenario,
                seed: int = 0) -> dict[str, np.ndarray]:
    """Kontrolcüyü senaryoda çalıştırır; k. eleman k. adımın sonucudur."""
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    controller.reset()
    dt = env.unwrapped.dt
    keys = ("delta_f_hz", "p_ess", "u_ess", "soc", "p_diesel", "d_net")
    rec: dict[str, list] = {k: [] for k in keys}
    rec["shield_active"] = []
    rec["reward"] = []
    done = False
    while not done:
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        for k in keys:
            rec[k].append(info[k])
        rec["shield_active"].append(bool(info["shield_active"]))
        # Görev ödülü: kalkan cezası hariç, böylece mean_cost tüm kontrolcülerde aynı büyüklüktür
        rec["reward"].append(float(info.get("reward_task", reward)))
        done = bool(terminated or truncated)
    traj = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    traj["shield_active"] = np.asarray(rec["shield_active"], dtype=bool)
    traj["t"] = (np.arange(len(rec["reward"])) + 1) * dt
    return traj
