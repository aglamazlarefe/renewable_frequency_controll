import os
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from envs.microgrid_lfc_env import MicrogridLFCEnv

def train_model():
    # Ortamı başlat
    env = MicrogridLFCEnv(max_steps=2000)
    
    # Sürekli eylem uzayı için gürültü (Exploration için gerekli)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # DDPG Modelini tanımla
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, 
                 learning_rate=1e-3, buffer_size=100000, batch_size=128, device="cuda")
    
    print("Eğitim başlıyor...")
    # Eğitim adımı (Proje için 50,000 - 100,000 arası önerilir)
    model.learn(total_timesteps=50000, log_interval=10)
    
    # Modeli kaydet
    os.makedirs("models", exist_ok=True)
    model.save("models/ddpg_microgrid_model")
    print("Model başarıyla kaydedildi: models/ddpg_microgrid_model.zip")

if __name__ == "__main__":
    train_model()