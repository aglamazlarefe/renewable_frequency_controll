import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.Data_loader import load_epias_wind_data

class MicrogridLFCEnv(gym.Env):
    def __init__(self, data_path="data/epias_wind_data.csv", max_steps=1000):
        super(MicrogridLFCEnv, self).__init__()
        
        self.M = 4.0      
        self.D = 0.015    
        self.dt = 0.1     
        self.K_vir = 0.5  
        
        self.max_steps = max_steps
        self.current_step = 0
        
        self.wind_profile = load_epias_wind_data(data_path, self.max_steps)
        self.base_load = 0.6 
        
        high_obs = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.delta_f = 0.0
        self.prev_delta_f = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.delta_f = 0.0
        self.prev_delta_f = 0.0
        
        obs = np.array([self.delta_f, self.wind_profile[self.current_step]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        p_ess = action[0] 
        p_wind = self.wind_profile[self.current_step]
        p_load = self.base_load + np.random.normal(0, 0.02) 
        
        df_dt = (self.delta_f - self.prev_delta_f) / self.dt
        virtual_inertia_power = -self.K_vir * df_dt
        
        delta_p = (p_wind + p_ess + virtual_inertia_power) - p_load
        
        self.prev_delta_f = self.delta_f
        self.delta_f += ((delta_p - self.D * self.delta_f) / self.M) * self.dt
        
        # --- YAPAY ZEKA OPTİMİZASYONU ---
        # Eski ceza: 5.0 * p_ess**2
        # Yeni ceza: 20.0 * p_ess**2
        # Batarya kullanımına verilen cezayı 4 katına çıkardık. AI artık bataryayı kullanırken aşırı cimri davranacak.
        reward = - (10.0 * (self.delta_f ** 2) + 20.0 * (p_ess ** 2))
        
        if abs(self.delta_f) > 0.15: # Güvenlik sınırını daralttık
            reward -= 50.0 
            
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False
        
        obs = np.array([self.delta_f, p_wind], dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}