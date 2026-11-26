"""
train.py — TD3 / SAC eğitimi (CPU, SPEC §3.10)
==============================================

Kullanım:
    python train.py --algo td3 --seed 0 --steps 150000 [--no-shield] [--w-u X]
                    [--out-dir D] [--log-dir D]

Eğitim ve doğrulama ortamları yalnızca EĞİTİM bölmesinden senaryo üretir;
test bölmesi burada hiçbir zaman kullanılmaz.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from agents.shield import ShieldedEnv
from config import RewardWeights
from envs.microgrid_lfc_env import MicrogridLFCEnv
from envs.scenarios import default_generator, make_scenario_bank

# --- Eğitim betiğine özgü ayarlar (fiziksel sabit değil) ---
ALGOS = {"td3": TD3, "sac": SAC}
EVAL_SEED = 12345            # doğrulama ortamının sabit tohumu
EVAL_FREQ_STEPS = 10_000     # EvalCallback aralığı [adım]
EVAL_EPISODES = 5            # doğrulamada bölüm sayısı
TD3_NOISE_SIGMA = 0.1
DEFAULTS = dict(
    policy_kwargs=dict(net_arch=[256, 256]),
    train_freq=1,
    gradient_steps=1,
    learning_starts=5000,
    batch_size=256,
    buffer_size=300_000,
)


class SeedOnFirstReset(gym.Wrapper):
    """Ortamı ilk reset'te verilen tohumla tohumlar; sonraki reset'ler akışı sürdürür."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._pending_seed: int | None = seed

    def reset(self, *, seed=None, options=None):
        if seed is None and self._pending_seed is not None:
            seed = self._pending_seed
        self._pending_seed = None
        return self.env.reset(seed=seed, options=options)


class CycleScenarios(gym.Wrapper):
    """Her reset'te sabit bir bankadan sıradaki senaryoyu verir.

    Doğrulama ortamı her değerlendirmede aynı senaryoları görsün diye kullanılır. SB3'ün
    evaluate_policy'si değerlendirme başına 1 açık reset + her bölüm sonunda 1 otomatik reset yapar
    (n_eval_episodes + 1 reset); banka bu uzunlukta tutulursa her değerlendirme banka[0:n] ile başlar.
    """

    def __init__(self, env: gym.Env, scenarios: list):
        super().__init__(env)
        self._bank = list(scenarios)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        scenario = self._bank[self._i % len(self._bank)]
        self._i += 1
        return self.env.reset(seed=seed, options={**(options or {}), "scenario": scenario})


def make_eval_env(shielded: bool = True, reward: RewardWeights | None = None) -> gym.Env:
    """Eğitim bölmesinden sabit senaryolu doğrulama ortamı (test bölmesi kullanılmaz)."""
    bank = make_scenario_bank(default_generator("train").wind_levels, EVAL_EPISODES + 1, EVAL_SEED)
    env: gym.Env = MicrogridLFCEnv(scenario_generator=default_generator("train"),
                                   reward=reward if reward is not None else RewardWeights())
    if shielded:
        env = ShieldedEnv(env)
    return Monitor(CycleScenarios(env, bank))


def make_env(seed: int, shielded: bool = True, split: str = "train",
             reward: RewardWeights | None = None) -> gym.Env:
    """Monitor(ShieldedEnv(MicrogridLFCEnv)) veya Monitor(MicrogridLFCEnv); ilk reset'te tohumlanır."""
    env: gym.Env = MicrogridLFCEnv(
        scenario_generator=default_generator(split),
        reward=reward if reward is not None else RewardWeights(),
    )
    if shielded:
        env = ShieldedEnv(env)
    env = SeedOnFirstReset(env, seed)
    return Monitor(env)


class EpisodeMetricsCallback(BaseCallback):
    """Bölüm sonunda ep_rmse_hz, ep_autonomy, ep_throughput_pu_s değerlerini kaydeder."""

    KEYS = ("ep_rmse_hz", "ep_autonomy", "ep_throughput_pu_s")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for key in self.KEYS:
                if key in info:
                    self.logger.record(f"episode/{key}", float(info[key]))
        return True


def _model_name(algo: str, seed: int, shielded: bool, w_u: float | None) -> str:
    name = f"{algo}_seed{seed}" if shielded else f"{algo}_noshield_seed{seed}"
    if w_u is not None and w_u != RewardWeights().w_u:
        name += f"_wu{w_u:g}"
    return name


def train(algo: str, seed: int, total_steps: int, out_dir: str | Path = "models",
          log_dir: str | Path | None = "logs", shielded: bool = True,
          w_u: float | None = None, **overrides) -> Path:
    """Modeli eğitir, `out_dir/{ad}.zip` olarak kaydeder ve yolunu döndürür."""
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Bilinmeyen algoritma: {algo!r}; beklenen: {sorted(ALGOS)}")

    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = _model_name(algo, seed, shielded, w_u)
    reward = replace(RewardWeights(), w_u=w_u) if w_u is not None else RewardWeights()

    env = make_env(seed, shielded, "train", reward)
    kwargs = deepcopy(DEFAULTS)  # SB3 policy_kwargs sözlüğünü değiştirir; paylaşılan varsayılan kirlenmesin
    kwargs.update(overrides)
    if algo == "td3" and "action_noise" not in kwargs:
        n_act = env.action_space.shape[0]
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_act), sigma=TD3_NOISE_SIGMA * np.ones(n_act))
    if log_dir is not None:
        kwargs["tensorboard_log"] = str(log_dir)

    model = ALGOS[algo]("MlpPolicy", env, seed=seed, device="auto", verbose=0, **kwargs)

    callbacks = []
    if log_dir is not None:
        eval_env = make_eval_env(shielded, reward)
        callbacks.append(EpisodeMetricsCallback())
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path=str(out_dir / "best" / name),
            log_path=str(Path(log_dir) / name), eval_freq=EVAL_FREQ_STEPS,
            n_eval_episodes=EVAL_EPISODES, deterministic=True))

    target = out_dir / f"{name}.zip"
    try:
        model.learn(total_timesteps=total_steps, callback=CallbackList(callbacks) if callbacks else None,
                    tb_log_name=name, progress_bar=False)
    except KeyboardInterrupt:
        model.save(str(out_dir / f"{name}_interrupted.zip"))
        raise
    model.save(str(target))
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="TD3/SAC eğitimi")
    p.add_argument("--algo", choices=sorted(ALGOS), default="td3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--no-shield", action="store_true")
    p.add_argument("--w-u", type=float, default=None)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    t0 = time.perf_counter()
    path = train(args.algo, args.seed, args.steps, out_dir=args.out_dir, log_dir=args.log_dir,
                 shielded=not args.no_shield, w_u=args.w_u)
    sps = args.steps / (time.perf_counter() - t0)
    print(f"Kaydedildi: {path}")
    print(f"Hız: {sps:.1f} adım/s")


if __name__ == "__main__":
    main()
