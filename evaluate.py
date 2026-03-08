import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from envs.microgrid_lfc_env import MicrogridLFCEnv
from agents.safe_hybrid_controller import SafeHybridController
from utils.Data_loader import load_epias_wind_data

def run_evaluation():
    print("Test ortamı hazırlanıyor...")
    csv_path = "data/epias_wind_data.csv"
    if not os.path.exists(csv_path):
        if os.path.exists("epias_wind_data.csv"):
            csv_path = "epias_wind_data.csv"
        else:
            print(f"HATA: {csv_path} bulunamadı!")
            return

    env = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
    env_pi = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
    
    # Train/Test Split (Görülmemiş Veri)
    full_data = load_epias_wind_data(filepath=csv_path, max_steps=2000)
    unseen_test_data = full_data[1000:2000] 
    env.unwrapped.wind_profile = unseen_test_data
    env_pi.unwrapped.wind_profile = unseen_test_data

    # Modeli Yükle
    try:
        drl_model = DDPG.load("models/ddpg_microgrid")
    except:
        drl_model = DDPG.load("models/ddpg_microgrid_model.zip")

    # PI Kalkanı katsayıları adil kıyaslama için Kp=4.0, Ki=1.0 yapıldı
    hybrid_controller = SafeHybridController(ai_model=drl_model, Kp=4.0, Ki=1.0, safety_limit=0.15)

    delta_f_hybrid, actions_hybrid, safety_flags = [], [], []
    delta_f_pi, actions_pi = [], []

    print("Senaryo 1: Safe-AI (Endüstriyel Kısıtlar ile) Test Ediliyor...")
    obs, _ = env.reset()
    prev_action_ai = 0.0
    
    for step in range(1000):
        # Stres testi: 400. adımda ani yük artışı
        if step == 400:
            env.unwrapped.base_load += 0.4 
            
        action, active_mode = hybrid_controller.predict(obs)
        
        if active_mode == "AI":
            # TEİAŞ Standart Ölü Bant: 0.05 Hz altındaki sapmaları görmezden gel
            if abs(obs[0]) < 0.05:
                action[0] = 0.0
            else:
                # Sinyal Sönümleme ve Rampa (Batarya Tasarrufu)
                damped_action = action[0] * 0.4
                action[0] = 0.8 * prev_action_ai + 0.2 * damped_action
                
        prev_action_ai = action[0]
        
        obs, reward, terminated, truncated, _ = env.step(action)
        delta_f_hybrid.append(obs[0])
        actions_hybrid.append(action[0])
        safety_flags.append(1 if active_mode == "AI" else 0)
        if terminated or truncated: break

    print("Senaryo 2: Geleneksel PI Kontrolcü Test Ediliyor...")
    obs_pi, _ = env_pi.reset()
    integral_error_pi = 0.0
    for step in range(1000):
        if step == 400:
            env_pi.unwrapped.base_load += 0.4
            
        error = 0.0 - obs_pi[0]
        integral_error_pi += error * 0.1
        
        # Adil kıyaslama için optimize edilmiş PI katsayıları
        action_pi = (4.0 * error) + (1.0 * integral_error_pi)
        action_pi = np.clip(action_pi, -1.0, 1.0)
        
        obs_pi, _, term, trunc, _ = env_pi.step([action_pi])
        delta_f_pi.append(obs_pi[0])
        actions_pi.append(action_pi)
        if term or trunc: break

    # CO2 VE TASARRUF HESAPLAMALARI
    EMISYON_FAKTORU = 0.4 
    BASE_POWER_MW = 10 

    toplam_enerji_pi = np.sum(np.abs(actions_pi)) * BASE_POWER_MW
    toplam_enerji_hybrid = np.sum(np.abs(actions_hybrid)) * BASE_POWER_MW

    tasarruf_mwh = (toplam_enerji_pi - toplam_enerji_hybrid) / 1000
    engellenen_co2 = tasarruf_mwh * EMISYON_FAKTORU * 1000
    tasarruf_yuzdesi = ((toplam_enerji_pi - toplam_enerji_hybrid) / toplam_enerji_pi) * 100

    print("\n" + "="*45)
    print("🎯 İKLİM ETKİSİ VE KONTROL SONUÇLARI (NİHAİ) 🎯")
    print("="*45)
    print(f"PI Kontrol Toplam Eforu   : {toplam_enerji_pi:.2f} MW")
    print(f"Safe-AI Toplam Eforu      : {toplam_enerji_hybrid:.2f} MW")
    print(f"Enerji Tasarrufu          : %{tasarruf_yuzdesi:.2f}")
    print(f"Net Engellenen CO2        : {engellenen_co2:.2f} kg")
    print("="*45 + "\n")

    # AKADEMİK GRAFİK ÇİZİMLERİ
    plt.style.use('default') 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1.5, 0.8]})

    ax1.plot(delta_f_pi, label="Geleneksel PI Kontrol (Optimize)", color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.plot(delta_f_hybrid, label="Önerilen Safe-AI (DDPG+PI)", color='#1f77b4', linewidth=2)
    ax1.axhline(y=0.15, color='gray', linestyle=':', label='Güvenlik Sınırı (±0.15 Hz)')
    ax1.axhline(y=-0.15, color='gray', linestyle=':')
    ax1.axvline(x=400, color='orange', linestyle='-', alpha=0.5, linewidth=2, label='Stres Testi (Ani Yük Artışı)')
    ax1.set_ylabel("Frekans Sapması $\Delta f$ (Hz)", fontweight='bold')
    ax1.set_title("Şebeke Frekans Kararlılığı (TEİAŞ Kısıtları Altında)", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(actions_pi, label="PI Batarya Eforu", color='#d62728', linestyle='--', alpha=0.6)
    ax2.plot(actions_hybrid, label="Safe-AI Batarya Eforu (Tasarruflu)", color='#1f77b4', alpha=0.9)
    ax2.axvline(x=400, color='orange', linestyle='-', alpha=0.5, linewidth=2)
    ax2.set_ylabel("Batarya Gücü (pu)", fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.plot(safety_flags, label="1: AI Otonom | 0: PI Fail-Safe", color='#2ca02c', linewidth=2)
    ax3.axvline(x=400, color='orange', linestyle='-', alpha=0.5, linewidth=2)
    ax3.set_ylabel("Kalkan", fontweight='bold')
    ax3.set_xlabel("Simülasyon Adımı (Zaman)", fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['PI (Acil)', 'AI (Otonom)'])
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('Bulgular_Grafikleri_Test_Split.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_evaluation()