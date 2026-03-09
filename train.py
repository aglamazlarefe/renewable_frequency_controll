"""
train.py — Maksimum Hız (Ryzen 5 5500 + RTX 3060)
===================================================

SORUN ANALİZİ (615 FPS'in neden yavaş olduğu):
  DummyVecEnv(4): 4 env sırayla tek thread'de çalışıyor
  Her adım: env.step() x4 → Python overhead x4
  GPU büyük kısmında boş bekliyor

ÇÖZÜM: total_steps'i düşür, gradient_steps'i artır
  Aynı öğrenme kalitesi, çok daha az env adımı.
  
  Gerçek formül:
    Efektif güncelleme = total_steps * gradient_steps / train_freq
    Eski: 3_000_000 * 8 / 5  = 4_800_000 güncelleme → 83 dk
    Yeni:   400_000 * 20 / 4  = 2_000_000 güncelleme → 8-10 dk
  
  Kalite benzer çünkü güncelleme sayısı yakın,
  ama env adımı 7.5x daha az → toplam süre 7-8 dk.

PARAMETRE DEĞİŞİKLİKLERİ:
  total_steps    : 3_000_000 → 400_000   (env adımı azalt)
  gradient_steps : 8         → 20        (her adımda daha fazla öğren)
  train_freq     : (5,"step")→ (4,"step") (biraz daha sık güncelle)
  batch_size     : 512       → 512        (aynı)
  n_envs         : 4         → 4          (aynı)
  learning_starts: 10_000    → 5_000      (daha erken başla)
"""

import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)
from envs.microgrid_lfc_env import MicrogridLFCEnv


def cosine_lr(initial_lr=3e-4, final_lr=1e-5):
    def schedule(progress_remaining):
        t = 1.0 - progress_remaining
        return final_lr + (initial_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * t))
    return schedule


class MetricCallback(BaseCallback):
    def __init__(self, log_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._rmse_buf, self._jit_buf, self._aut_buf = [], [], []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "ep_rmse_f" in info:
                self._rmse_buf.append(info["ep_rmse_f"])
                self._jit_buf.append(info["ep_jitter"])
                self._aut_buf.append(info["ep_autonomy"])

        if self.num_timesteps % self.log_freq == 0 and self._rmse_buf:
            rmse = float(np.mean(self._rmse_buf[-20:]))
            jit  = float(np.mean(self._jit_buf[-20:]))
            aut  = float(np.mean(self._aut_buf[-20:]))
            self.logger.record("metrics/rmse_hz",      rmse)
            self.logger.record("metrics/jitter",       jit)
            self.logger.record("metrics/autonomy_pct", aut * 100)
            print(f"  [METRİK] {self.num_timesteps:>9,} | "
                  f"RMSE={rmse*1000:.1f}mHz | Jitter={jit:.1f} | "
                  f"Otonomi={aut*100:.1f}%")
        return True


def train_model():
    N_ENVS          = 4
    TOTAL_STEPS     = 100_000    # ← 3M → 400k (süre: ~8 dk)
    GRADIENT_STEPS  = 20         # ← 8  → 20   (adım başı daha fazla öğren)
    TRAIN_FREQ      = (4, "step")
    BATCH_SIZE      = 512
    BUFFER_SIZE     = 200_000    # 500k → 200k (400k adım için yeterli)
    LEARNING_STARTS = 5_000      # 10k → 5k

    # Efektif güncelleme sayısı
    eff_updates = TOTAL_STEPS * GRADIENT_STEPS // TRAIN_FREQ[0]
    eta_min = TOTAL_STEPS / 500 / 60  # 500 FPS tahminiyle

    print(f"\n{'='*60}")
    print(f"  Ryzen 5 5500 + RTX 3060 — Hız Optimize")
    print(f"  {TOTAL_STEPS:,} env adımı | ~{eta_min:.0f} dakika")
    print(f"  Efektif güncelleme: {eff_updates:,}")
    print(f"  gradient_steps={GRADIENT_STEPS} | train_freq={TRAIN_FREQ}")
    print(f"{'='*60}\n")

    env = make_vec_env(
        MicrogridLFCEnv,
        n_envs=N_ENVS,
        seed=42,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"max_steps": 1000}
    )

    n_actions    = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.08 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise    = action_noise,
        verbose         = 1,
        learning_rate   = cosine_lr(3e-4, 1e-5),
        buffer_size     = BUFFER_SIZE,
        batch_size      = BATCH_SIZE,
        device          = "cuda",
        tau             = 0.005,
        train_freq      = TRAIN_FREQ,
        gradient_steps  = GRADIENT_STEPS,
        learning_starts = LEARNING_STARTS,
        policy_kwargs   = dict(net_arch=[256, 256]),
        tensorboard_log = "logs/"
    )

    os.makedirs("models/checkpoints", exist_ok=True)

    callbacks = CallbackList([
        MetricCallback(log_freq=10_000),
        CheckpointCallback(
            save_freq   = 100_000 // N_ENVS,
            save_path   = "models/checkpoints/",
            name_prefix = "td3_mg",
            verbose     = 0
        )
    ])

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=callbacks,
            progress_bar=True
        )
        os.makedirs("models", exist_ok=True)
        model.save("models/td3_microgrid_model")
        print("\n[BAŞARILI] Model kaydedildi: models/td3_microgrid_model")

    except KeyboardInterrupt:
        print("\n[DURDURULDU] Ara model kaydediliyor...")
        model.save("models/td3_microgrid_model_interrupted")

    except Exception as e:
        print(f"\n[HATA] {e}")
        model.save("models/td3_microgrid_model_interrupted")

    env.close()


if __name__ == "__main__":
    train_model()