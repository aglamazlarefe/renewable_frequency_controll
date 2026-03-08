import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.microgrid_lfc_env import MicrogridLFCEnv

def train_model():
    # 1. PARALELLEŞTİRME: Ortamı 8 paralel kopyaya bölüyoruz (RTX 3060 için ideal)
    # Bu, 1 gerçek saniyede 8 saniyelik veri toplamanı sağlar.
    n_envs = 8
    print(f"{n_envs} paralel ortam oluşturuluyor...")
    
    env = make_vec_env(
        MicrogridLFCEnv, 
        n_envs=n_envs, 
        seed=42, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"max_steps": 1000}
    )
    
    # 2. HIZLI YAKINSAMA İÇİN HİPERPARAMETRELER
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Batch size'ı 512 yaparak GPU çekirdeklerini tam dolduruyoruz.
    # Learning rate'i hafifçe artırarak (5e-4) öğrenmeyi hızlandırıyoruz.
    model = TD3(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        verbose=1, 
        learning_rate=5e-4, 
        buffer_size=100000, 
        batch_size=512, 
        device="cuda",
        train_freq=(1, "step"),
        gradient_steps=1,
        learning_starts=5000, # Buffer dolana kadar hızlı başla
        tau=0.001
    )
    
    # 3. EĞİTİM
    # 300.000 adım / 8 env = Her env başına 37.500 adım. Çok daha hızlı bitecek.
    total_steps = 300000
    print(f"Eğitim başlıyor: {total_steps} toplam adım (Paralel)... hedef < 15 dk.")
    
    model.learn(total_timesteps=total_steps, progress_bar=True)
    
    # 4. KAYDET
    os.makedirs("models", exist_ok=True)
    model.save("models/td3_microgrid_model")
    print("Model başarıyla kaydedildi: models/td3_microgrid_model.zip")

if __name__ == "__main__":
    train_model()