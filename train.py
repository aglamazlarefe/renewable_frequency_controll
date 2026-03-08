import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.microgrid_lfc_env import MicrogridLFCEnv

def train_model():
    n_envs = 8
    env = make_vec_env(MicrogridLFCEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    
    model = TD3(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        verbose=1, 
        learning_rate=3e-4, 
        buffer_size=150000, 
        batch_size=512,      # Batch size'ı 256 yaparak gradyan gürültüsünü azalttık
        device="cuda",
        tau=0.001,           # Target network güncellemelerini yavaşlattık
        train_freq=(1, "step"),    
        gradient_steps=1,     
        learning_starts=5000 
    )
    
    print("Stabil Eğitim Başlıyor... Critic Loss kontrol ediliyor.")
    model.learn(total_timesteps=300000, progress_bar=True)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/td3_microgrid_model")
    print("Stabil Model Kaydedildi.")

if __name__ == "__main__":
    train_model()