import gymnasium as gym
from gymnasium import spaces
import numpy as np
From Utils.data_loader import load_epias_wind_data

class MicrogridLFCEnv(gym.Env):
    def __init__(self, data_path="data/epias_wind_data.csv", max_steps=1000):
        super(MicrogridLFCEnv, self).__init__()
        
        # Şebeke Parametreleri (Düşük ataletli sistem)
        self.M = 4.0      # Atalet sabiti (Inertia - Klasik sistemlere göre düşük)
        self.D = 0.015    # Yük sönümleme katsayısı
        self.dt = 0.1     # Simülasyon zaman adımı
        self.K_vir = 0.5  # Sanal atalet katsayısı (Virtual Inertia gain)
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # EPİAŞ Gerçek Rüzgar Verisini Yükle
        self.wind_profile = load_epias_wind_data(data_path, self.max_steps)
        
        # Sabit Yük (Örnek olarak 0.6 pu)
        self.base_load = 0.6 
        
        # Durum Uzayı: [delta_f, delta_P_wind]
        high_obs = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        
        # Eylem Uzayı: Enerji Depolama Sistemi (ESS) çıkış gücü [-1.0, 1.0]
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
        p_ess = action[0] # Yapay zekanın (Bataryanın) şebekeye verdiği/çektiği güç
        
        # Anlık üretim ve tüketim
        p_wind = self.wind_profile[self.current_step]
        p_load = self.base_load + np.random.normal(0, 0.02) # Yükte ufak gürültüler
        
        # Sanal Atalet Etkisi: Frekans türevine tepki
        df_dt = (self.delta_f - self.prev_delta_f) / self.dt
        virtual_inertia_power = -self.K_vir * df_dt
        
        # Net güç değişimi
        delta_p = (p_wind + p_ess + virtual_inertia_power) - p_load
        
        # Salınım Denklemi (Swing Equation) ile yeni frekans sapmasını hesapla
        self.prev_delta_f = self.delta_f
        self.delta_f += ((delta_p - self.D * self.delta_f) / self.M) * self.dt
        
        # Ödül Fonksiyonu (Ceza tasarımı)
        # Frekans sapmasını cezalandır, aşırı güç kullanımını (action) cezalandır
        reward = - (10.0 * (self.delta_f ** 2) + 0.1 * (p_ess ** 2))
        
        # Güvenlik sınırı ihlali için devasa ceza (Safe RL)
        if abs(self.delta_f) > 0.2:
            reward -= 50.0 
            
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False
        
        obs = np.array([self.delta_f, p_wind], dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}