import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.microgrid_lfc_env import MicrogridLFCEnv

def train_model():
    n_envs = 8
    env = make_vec_env(
        MicrogridLFCEnv, 
        n_envs=n_envs, 
        seed=42, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"max_steps": 1000}
    )
    
    n_actions = env.action_space.shape[-1]
    # Gürültü seviyesini azalttık. Titremeyi eğitim aşamasında da baskılıyoruz.
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    
    model = TD3(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        verbose=1, 
        learning_rate=1e-4, # Daha yavaş ve sağlam adımlar
        buffer_size=200000, 
        batch_size=512,      
        device="cuda",
        tau=0.001,           
        train_freq=(1, "step"),
        gradient_steps=1,
        learning_starts=10000,
        policy_kwargs=dict(net_arch=[400, 300])
    )
    
    # 400.000 adım yeterlidir. Önemli olan kalite, miktar değil.
    total_steps = 300000
    print(f"Hassas Eğitim Başlıyor: {total_steps} adım...")
    
    try:
        model.learn(total_timesteps=total_steps, progress_bar=True)
        os.makedirs("models", exist_ok=True)
        model.save("models/td3_microgrid_model")
        print("\n[BAŞARILI] Pürüzsüz kontrol modeli kaydedildi.")
    except Exception as e:
        print(f"\n[HATA] {e}")

if __name__ == "__main__":
    train_model()