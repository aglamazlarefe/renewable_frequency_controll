import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.Data_loader import load_epias_wind_data

class MicrogridLFCEnv(gym.Env):
    def __init__(self, data_path="data/epias_wind_data.csv", max_steps=1000):
        super(MicrogridLFCEnv, self).__init__()

        # ── Fiziksel Sabitler ──────────────────────────────────────
        self.M     = 4.0   
        self.D     = 0.8   
        self.dt    = 0.1
        self.K_vir = 0.5

        self.max_steps         = max_steps
        self.current_step      = 0
        self.base_load_default = 0.6

        self.wind_profile = load_epias_wind_data(data_path, self.max_steps)

        # Gözlem: [delta_f_norm, wind_power, prev_p_ess]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.delta_f      = 0.0
        self.prev_delta_f = 0.0
        self.prev_p_ess   = 0.0
        self.base_load    = self.base_load_default

        # PI Kalkan parametreleri
        self.KP_PI       = 20.0
        self.KI_PI       = 8.0
        self.pi_integral = 0.0
        self.SHIELD_THR  = 0.15  # Hz

        self._ep_f_sq_sum     = 0.0
        self._ep_jitter       = 0.0
        self._ep_shield_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.delta_f      = 0.0
        self.prev_delta_f = 0.0
        self.prev_p_ess   = 0.0
        self.pi_integral  = 0.0
        self.base_load    = self.base_load_default + np.random.uniform(-0.05, 0.05)

        self._ep_f_sq_sum     = 0.0
        self._ep_jitter       = 0.0
        self._ep_shield_steps = 0

        return self._get_obs(), {}

    def step(self, action):
        raw_p_ess = float(action[0])

        # 1) SLEW RATE — 0.015 yetersizdi, 0.025'e çıkarıldı (Çeviklik artırıldı)
        diff  = raw_p_ess - self.prev_p_ess
        p_cmd = self.prev_p_ess + np.clip(diff, -0.025, 0.025)
        p_cmd = float(np.clip(p_cmd, -1.0, 1.0))

        # 2) SAFETY SHIELD
        shield_active = False
        if abs(self.delta_f) > self.SHIELD_THR:
            shield_active = True
            self._ep_shield_steps += 1
            error             = -self.delta_f
            self.pi_integral += error * self.dt
            self.pi_integral  = np.clip(self.pi_integral, -0.5, 0.5)
            p_ess = float(np.clip(
                self.KP_PI * error + self.KI_PI * self.pi_integral, -1.0, 1.0
            ))
        else:
            p_ess = p_cmd
            # Bumpless transfer: AI devralırken integral yavaşça söner
            self.pi_integral *= 0.90 

        # 3) Rüzgâr ve yük
        step_idx  = min(self.current_step, self.max_steps - 1)
        p_wind    = float(self.wind_profile[step_idx])
        p_load    = self.base_load + np.random.normal(0, 0.002)

        # 4) Swing denklemi
        df_dt     = (self.delta_f - self.prev_delta_f) / self.dt
        p_virtual = -self.K_vir * df_dt
        delta_p   = (p_wind + p_ess + p_virtual) - p_load

        self.prev_delta_f = self.delta_f
        self.delta_f = float(np.clip(
            self.delta_f + ((delta_p - self.D * self.delta_f) / self.M) * self.dt,
            -0.5, 0.5
        ))

        # 5) ÖDÜL DENGESİ (REVISED)
        # Bonus: Otonomiye ciddi puan ver
        bonus     = 5.0 if not shield_active else -2.0
        
        # Frekans Cezası: 25.0 -> 120.0 (Ajan frekans hatasını ciddiye almalı)
        f_penalty = 120.0 * (self.delta_f ** 2)
        
        # Enerji Cezası: Düşürüldü, ajan bataryayı kullanmaktan korkmamalı
        e_penalty = 5.0 * (p_ess ** 2)
        
        # Jitter Cezası: Shield aktifse ceza kesilmez (Ajanın suçu değil)
        # Slew rate arttığı için katsayı 500 -> 350 yapıldı
        j_penalty = 350.0 * (p_ess - self.prev_p_ess) ** 2 if not shield_active else 0.0
        
        barrier   = 400.0 * max(0, abs(self.delta_f) - 0.11) ** 2

        total_reward = bonus - (f_penalty + e_penalty + j_penalty + barrier)

        # 6) Metrik güncelle
        self._ep_f_sq_sum += self.delta_f ** 2
        self._ep_jitter   += abs(p_ess - self.prev_p_ess)
        self.prev_p_ess    = p_ess
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)

        info = {
            "p_ess":         p_ess,
            "delta_f":       self.delta_f,
            "shield_active": shield_active,
        }
        if terminated:
            n = max(self.current_step, 1)
            info["ep_rmse_f"]     = float(np.sqrt(self._ep_f_sq_sum / n))
            info["ep_jitter"]     = float(self._ep_jitter)
            info["ep_autonomy"]   = 1.0 - self._ep_shield_steps / n

        return self._get_obs(), total_reward, terminated, False, info

    def _get_obs(self):
        step_idx     = min(self.current_step, self.max_steps - 1)
        # Ölçekleme: Ajan küçük sapmaları (20-30 mHz) daha net görsün
        delta_f_norm = float(np.clip(self.delta_f / 0.2, -1.0, 1.0))
        return np.array([
            delta_f_norm,
            float(self.wind_profile[step_idx]),
            float(self.prev_p_ess)
        ], dtype=np.float32)