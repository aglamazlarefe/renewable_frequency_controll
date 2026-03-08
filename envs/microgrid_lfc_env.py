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
        self.prev_p_ess = 0.0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.delta_f = 0.0
        self.prev_delta_f = 0.0
        self.prev_p_ess = 0.0
        obs = np.array([self.delta_f, self.wind_profile[self.current_step]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        raw_p_ess = action[0]
        
        # --- DONANIM KORUMA (Slew Rate Limit) ---
        # Jitter'ı engellemek için rampa sınırını 0.1'den 0.05'e çektik
        diff = raw_p_ess - self.prev_p_ess
        p_ess = self.prev_p_ess + np.clip(diff, -0.05, 0.05)
        
        # Deadband: 0.03 Hz altında bataryayı yorma (Enerji tasarrufu)
        if abs(self.delta_f) < 0.03:
            p_ess = 0.0
            
        p_ess = p_ess * 0.8 # Sönümlemeyi gevşettik (0.7 -> 0.8)

        p_wind = self.wind_profile[self.current_step]
        p_load = self.base_load + np.random.normal(0, 0.01) 
        
        df_dt = (self.delta_f - self.prev_delta_f) / self.dt
        virtual_inertia_power = -self.K_vir * df_dt
        delta_p = (p_wind + p_ess + virtual_inertia_power) - p_load
        
        self.prev_delta_f = self.delta_f
        self.delta_f += ((delta_p - self.D * self.delta_f) / self.M) * self.dt
        
        # --- ÖDÜL FONKSİYONU REVİZYONU ---
        # Katsayıları normalleştiriyoruz (Loss patlamasını önlemek için)
        
        # 1. Frekans Hatası: Katsayıyı 15.0'da sabitle
        f_penalty = 15.0 * (self.delta_f ** 2)
        
        # 2. Enerji Cezası: Çok yüksek (40) olması ajanı korkutur, 20.0 idealdir
        e_penalty = 20.0 * (p_ess ** 2)
        
        # 3. Jitter Cezası: 30 çok fazlaydı, 5.0 ile başlayıp ajanın pürüzsüzleşmesini bekleyelim
        change_penalty = 5.0 * (p_ess - self.prev_p_ess)**2
        
        # 4. Shield / Kalkan Cezası: -150 yerine -10.0 gibi daha yumuşak bir ceza verelim
        # Büyük cezalar ajanı "donup kalmaya" iter.
        shield_penalty = 10.0 if abs(self.delta_f) > 0.12 else 0
        
        reward = -(f_penalty + e_penalty + change_penalty + shield_penalty)
            
        self.prev_p_ess = p_ess 
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        
        obs = np.array([self.delta_f, p_wind], dtype=np.float32)
        return obs, reward, terminated, False, {}