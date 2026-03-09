import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.Data_loader import load_epias_wind_data

class MicrogridLFCEnv(gym.Env):
    def __init__(self, data_path="data/epias_wind_data.csv", max_steps=1000):
        super(MicrogridLFCEnv, self).__init__()
        # Fiziksel Sabitler
        self.M, self.D, self.dt, self.K_vir = 4.0, 0.015, 0.1, 0.5
        self.max_steps = max_steps
        self.current_step = 0
        self.wind_profile = load_epias_wind_data(data_path, self.max_steps)
        self.base_load_default = 0.6 
        
        # Gözlem: [Frekans Sapması, Rüzgar Gücü, Önceki Batarya Gücü]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.delta_f = 0.0
        self.prev_delta_f = 0.0
        self.prev_p_ess = 0.0 
        self.base_load = 0.6

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step, self.delta_f, self.prev_delta_f, self.prev_p_ess = 0, 0.0, 0.0, 0.0
        # Eğitimde ajanı zorlamak için yükü randomize ediyoruz
        self.base_load = self.base_load_default + np.random.uniform(-0.05, 0.05)
        return self._get_obs(), {}

    def _get_obs(self):
        step_idx = min(self.current_step, self.max_steps - 1)
        return np.array([
            np.float32(self.delta_f), 
            np.float32(self.wind_profile[step_idx]), 
            np.float32(self.prev_p_ess)
        ], dtype=np.float32)

    def step(self, action):
        raw_p_ess = action[0]
        
        # --- KRİTİK MÜDAHALE: MEKANİK RAMPA SINIRI (SLEW RATE) ---
        # Jitter'ı 10'un altına çekmek için değişimi %1 (0.01) ile sınırlıyoruz.
        # Bu, bataryanın 0'dan 1.0 güce çıkması için en az 10 saniye (100 adım) geçmesini zorunlu kılar.
        diff = raw_p_ess - self.prev_p_ess
        p_ess = self.prev_p_ess + np.clip(diff, -0.01, 0.01) 
        p_ess = p_ess * 0.92 # Hafif sönümleme

        p_wind = self.wind_profile[min(self.current_step, self.max_steps - 1)]
        p_load = self.base_load + np.random.normal(0, 0.002) 
        
        # Dinamik Denklem (Swing Equation)
        df_dt = (self.delta_f - self.prev_delta_f) / self.dt
        virtual_inertia_power = -self.K_vir * df_dt
        delta_p = (p_wind + p_ess + virtual_inertia_power) - p_load
        
        self.prev_delta_f = self.delta_f
        self.delta_f += ((delta_p - self.D * self.delta_f) / self.M) * self.dt
        
        # --- ÖDÜL SİSTEMİ (Vibrasyonu Öldüren Denge) ---
        # Otonomi bonusunu azalttık (Ajan hile yapmasın diye)
        reward = 1.0 
        
        f_penalty = 20.0 * (self.delta_f ** 2)
        e_penalty = 50.0 * (p_ess ** 2) 
        # Jitter cezası devasa artırıldı (1500). Titreme = Ölümcül Ceza.
        j_penalty = 1500.0 * (p_ess - self.prev_p_ess)**2
        
        # Bariyer: 0.10'dan itibaren ajanı 'fren yapmaya' zorlar
        barrier = 0
        if abs(self.delta_f) > 0.10:
            barrier = 400.0 * (abs(self.delta_f) - 0.10)**2
        
        total_reward = reward - (f_penalty + e_penalty + j_penalty + barrier)
            
        self.prev_p_ess = p_ess 
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        
        return self._get_obs(), total_reward, terminated, False, {}

