import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.Data_loader import load_epias_wind_data

class MicrogridLFCEnv(gym.Env):
    def __init__(self, data_path="data/epias_wind_data.csv", max_steps=1000):
        super(MicrogridLFCEnv, self).__init__()
        
        # Fiziksel Parametreler
        self.M, self.D, self.dt, self.K_vir = 4.0, 0.015, 0.1, 0.5
        self.max_steps = max_steps
        self.current_step = 0
        self.wind_profile = load_epias_wind_data(data_path, self.max_steps)
        self.base_load_default = 0.6 
        
        # Gözlem: [Frekans, Rüzgar, Önceki Güç]
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
        self.current_step = 0
        self.delta_f = 0.0
        self.prev_delta_f = 0.0
        self.prev_p_ess = 0.0
        
        # --- STRATEJİK İYİLEŞTİRME: Randomize Başlangıç ---
        # Ajan her uyandığında farklı bir yük seviyesiyle (0.5 - 0.7) karşılaşır.
        # Bu, ajanı her türlü senaryoya karşı 'bağışık' hale getirir.
        self.base_load = self.base_load_default + np.random.uniform(-0.1, 0.1)
        
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
        
        # --- DONANIM KORUMA (Slew Rate Limit) ---
        diff = raw_p_ess - self.prev_p_ess
        p_ess = self.prev_p_ess + np.clip(diff, -0.04, 0.04)
        p_ess = p_ess * 0.95 

        p_wind = self.wind_profile[min(self.current_step, self.max_steps - 1)]
        # Dinamik gürültü (Eğitimi zorlaştırır, ajanı güçlendirir)
        p_load = self.base_load + np.random.normal(0, 0.005) 
        
        df_dt = (self.delta_f - self.prev_delta_f) / self.dt
        virtual_inertia_power = -self.K_vir * df_dt
        delta_p = (p_wind + p_ess + virtual_inertia_power) - p_load
        
        self.prev_delta_f = self.delta_f
        self.delta_f += ((delta_p - self.D * self.delta_f) / self.M) * self.dt
        
        # --- OTONOMİ ODAKLI ÖDÜL SİSTEMİ ---
        # 1. Yaşama Bonusu (Otonomi Teşviki)
        reward = 3.0 
        
        # 2. Frekans ve Enerji Dengesi
        f_penalty = 12.0 * (self.delta_f ** 2)
        e_penalty = 25.0 * (p_ess ** 2) 
        j_penalty = 20.0 * (p_ess - self.prev_p_ess)**2
        
        # 3. Ölümcül Bariyer (Shield Sınırına Yakınlık)
        barrier = 0
        if abs(self.delta_f) > 0.12:
            barrier = 250.0 * (abs(self.delta_f) - 0.12)
        
        total_reward = reward - (f_penalty + e_penalty + j_penalty + barrier)
            
        self.prev_p_ess = p_ess 
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        
        return self._get_obs(), total_reward, terminated, False, {}