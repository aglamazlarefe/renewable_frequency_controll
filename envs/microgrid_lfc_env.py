import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.Data_loader import load_epias_wind_data

class MicrogridLFCEnv(gym.Env):
    def __init__(self, data_path="data/epias_wind_data.csv", max_steps=1000):
        super(MicrogridLFCEnv, self).__init__()
        
        self.M, self.D, self.dt, self.K_vir = 4.0, 0.015, 0.1, 0.5
        self.max_steps = max_steps
        self.current_step = 0
        self.wind_profile = load_epias_wind_data(data_path, self.max_steps)
        self.base_load = 0.6 
        
        self.observation_space = spaces.Box(-np.array([1, 1]), np.array([1, 1]), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.delta_f = 0.0
        self.prev_delta_f = 0.0
        self.prev_p_ess = 0.0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step, self.delta_f, self.prev_delta_f, self.prev_p_ess = 0, 0.0, 0.0, 0.0
        return np.array([self.delta_f, self.wind_profile[self.current_step]], dtype=np.float32), {}

    def step(self, action):
        raw_p_ess = action[0]
        
        # --- DONANIM KISITLARI ---
        # Jitter'ı bitirmek için rampa sınırını 0.03'e çekiyoruz (Çok katı kısıt)
        diff = raw_p_ess - self.prev_p_ess
        p_ess = self.prev_p_ess + np.clip(diff, -0.03, 0.03)
        
        # Sönümlemeyi %85'te tut (Ajanın kontrol gücü)
        p_ess = p_ess * 0.85 

        p_wind = self.wind_profile[self.current_step]
        p_load = self.base_load + np.random.normal(0, 0.01) 
        
        df_dt = (self.delta_f - self.prev_delta_f) / self.dt
        virtual_inertia_power = -self.K_vir * df_dt
        delta_p = (p_wind + p_ess + virtual_inertia_power) - p_load
        
        self.prev_delta_f = self.delta_f
        self.delta_f += ((delta_p - self.D * self.delta_f) / self.M) * self.dt
        
        # --- TASARRUF ODAKLI ÖDÜL SİSTEMİ ---
        # Frekans cezasını düşür, enerji ve değişim cezasını artır.
        f_penalty = 8.0 * (self.delta_f ** 2)
        e_penalty = 25.0 * (p_ess ** 2)      # Enerjiyi pahalı hale getirdik (Tasarruf için)
        j_penalty = 35.0 * (p_ess - self.prev_p_ess)**2 # Titremeyi çok pahalı yaptık (Pürüzsüzlük için)
        
        # Bariyer fonksiyonunu daha geniş tut (Otonomi için)
        barrier = 50.0 * (abs(self.delta_f) / 0.15)**2 if abs(self.delta_f) > 0.13 else 0
        
        reward = -(f_penalty + e_penalty + j_penalty + barrier)
            
        self.prev_p_ess = p_ess 
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        
        obs = np.array([self.delta_f, p_wind], dtype=np.float32)
        return obs, reward, terminated, False, {}