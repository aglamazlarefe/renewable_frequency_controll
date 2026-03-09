import os
import numpy as np
import time
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.microgrid_lfc_env import MicrogridLFCEnv

def train_model():
    # 1. PARALELLEŞTİRME
    n_envs = 8
    print(f"{n_envs} paralel ortam oluşturuluyor...")
    
    env = make_vec_env(
        MicrogridLFCEnv, 
        n_envs=n_envs, 
        seed=42, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"max_steps": 1000}
    )
    
    # 2. AKSİYON GÜRÜLTÜSÜ
    # Sigma'yı 0.1 tutarak ajanın yeni stratejiler aramaya devam etmesini sağlıyoruz.
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # 3. ULTRA DERİN AĞ YAPISI
    # Daha karmaşık non-lineer ilişkileri yakalamak için nöron sayılarını artırdık.
    policy_kwargs = dict(net_arch=[512, 512, 400, 300])
    
    # 4. TD3 ULTRA KONFİGÜRASYONU (6-8 SAATLİK STRATEJİ)
    # gradient_steps=10: GPU her adımda 10 kez güncelleme yapacak. Bu en ağır ama en kaliteli ayardır.
    # learning_rate=5e-5: Çok düşük LR ile hata payını minimize ediyoruz.
    model = TD3(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        verbose=1, 
        learning_rate=5e-5, 
        buffer_size=1000000, 
        batch_size=512,      
        device="cuda",
        policy_kwargs=policy_kwargs,
        tau=0.0005,          # Hedef ağ güncellemelerini iyice yavaşlatarak aşırı kararlılık sağlıyoruz
        train_freq=(1, "step"),
        gradient_steps=10,   # ULTRA YOĞUNLUK: Her adımda 10 güncelleme
        learning_starts=30000 # Isınma süresini 30k yaparak buffer'ın iyice dolmasını sağlıyoruz
    )
    
    print("\n" + "="*60)
    print("🌙 EFSANEVİ GECE EĞİTİMİ (ULTRA SÜRÜM) BAŞLIYOR 🌙")
    print("Hedef: Maksimum Otonomi, Minimum Jitter, Derin Kararlılık")
    print("Tahmini Süre: 6 - 8 Saat (RTX 3060)")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    try:
        # Toplam 3,000,000 adım. gradient_steps=10 ile 30 milyon gradyan güncellemesi!
        model.learn(total_timesteps=3000000, progress_bar=True)
        
        # 6. MODELİ KAYDET
        os.makedirs("models", exist_ok=True)
        model.save("models/td3_microgrid_model")
        
        end_time = time.time()
        duration = (end_time - start_time) / 3600
        print(f"\n[FİNAL] Eğitim {duration:.2f} saatte tamamlandı.")
        print("Model başarıyla kaydedildi: models/td3_microgrid_model.zip")
        
    except KeyboardInterrupt:
        print("\n[BİLGİ] Eğitim kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[HATA] Beklenmedik sorun: {e}")

if __name__ == "__main__":
    train_model()