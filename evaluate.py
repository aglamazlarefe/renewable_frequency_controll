import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs.microgrid_lfc_env import MicrogridLFCEnv
from agents.safe_hybrid_controller import SafeHybridController
from utils.Data_loader import load_epias_wind_data

def calculate_metrics(delta_f, actions, safety_flags):
    """Akademik raporlama için detaylı metrik hesaplamaları."""
    delta_f = np.array(delta_f)
    actions = np.array(actions)
    
    # 1. Kararlılık Metrikleri
    rmse = np.sqrt(np.mean(delta_f**2))
    nadir = np.min(delta_f)
    zenith = np.max(delta_f)
    max_dev = np.max(np.abs(delta_f))
    
    # 2. Verimlilik ve Efor
    total_effort = np.sum(np.abs(actions))
    # Kontrol Titremesi (Jitter/Smoothness): Düşük olması donanım sağlığı için iyidir
    jitter = np.sum(np.abs(np.diff(actions)))
    
    # 3. Güvenlik ve Otonomi
    ai_active_steps = np.sum(safety_flags)
    autonomy_ratio = (ai_active_steps / len(safety_flags)) * 100
    
    return {
        "rmse": rmse,
        "nadir": nadir,
        "zenith": zenith,
        "max_dev": max_dev,
        "effort": total_effort,
        "jitter": jitter,
        "autonomy": autonomy_ratio
    }

def run_evaluation():
    print("Test ortamı ve veriler hazırlanıyor...")
    csv_path = "data/epias_wind_data.csv"
    if not os.path.exists(csv_path):
        csv_path = "epias_wind_data.csv"

    env = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
    env_pi = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
    
    # Unseen Test Data
    full_data = load_epias_wind_data(filepath=csv_path, max_steps=2000)
    unseen_test_data = full_data[1000:2000] 
    env.unwrapped.wind_profile = unseen_test_data
    env_pi.unwrapped.wind_profile = unseen_test_data

    # Modeli Yükle
    model_path = "models/td3_microgrid_model"
    try:
        model = TD3.load(model_path)
    except:
        model = TD3.load(model_path + ".zip")

    # Hibrit Kontrolcü (Shielding)
    hybrid_controller = SafeHybridController(ai_model=model, Kp=4.0, Ki=1.0, safety_limit=0.15)

    delta_f_hybrid, actions_hybrid, safety_flags = [], [], []
    delta_f_pi, actions_pi = [], []

    # --- SENARYO 1: SAFE-AI ---
    print("Simülasyon 1: Safe-AI çalışıyor...")
    obs, _ = env.reset()
    for step in range(1000):
        if step == 400: env.unwrapped.base_load += 0.2
        action, active_mode = hybrid_controller.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        delta_f_hybrid.append(obs[0])
        actions_hybrid.append(action[0])
        safety_flags.append(1 if active_mode == "AI" else 0)
        if terminated or truncated: break

    # --- SENARYO 2: PI ---
    print("Simülasyon 2: PI çalışıyor...")
    obs_pi, _ = env_pi.reset()
    integral_error_pi = 0.0
    for step in range(1000):
        if step == 400: env_pi.unwrapped.base_load += 0.4
        error = 0.0 - obs_pi[0]
        integral_error_pi += error * 0.1
        action_pi = (4.0 * error) + (1.0 * integral_error_pi)
        action_pi = np.clip(action_pi, -1.0, 1.0)
        obs_pi, _, term, trunc, _ = env_pi.step([action_pi])
        delta_f_pi.append(obs_pi[0])
        actions_pi.append(action_pi)
        if term or trunc: break

    # --- HESAPLAMALAR ---
    m_ai = calculate_metrics(delta_f_hybrid, actions_hybrid, safety_flags)
    m_pi = calculate_metrics(delta_f_pi, actions_pi, [0]*len(actions_pi))

    BASE_POWER_MW = 10
    EMISYON_FAKTORU = 0.4

    toplam_mw_ai = m_ai['effort'] * BASE_POWER_MW
    toplam_mw_pi = m_pi['effort'] * BASE_POWER_MW
    tasarruf_yuzdesi = ((toplam_mw_pi - toplam_mw_ai) / toplam_mw_pi) * 100
    engellenen_co2 = (toplam_mw_pi - toplam_mw_ai) / 1000 * EMISYON_FAKTORU * 1000

    # --- DETAYLI RAPORLAMA ---
    print("\n" + "="*65)
    print("📊 KAPSAMLI MÜHENDİSLİK VE PERFORMANS ANALİZİ 📊")
    print("="*65)
    print(f"{'Metrik':<30} | {'Geleneksel PI':<15} | {'Önerilen Safe-AI':<15}")
    print("-" * 65)
    print(f"{'Frekans RMSE (Hz)':<30} | {m_pi['rmse']:.5f}         | {m_ai['rmse']:.5f}")
    print(f"{'Maksimum Sapma (Hz)':<30} | {m_pi['max_dev']:.4f}         | {m_ai['max_dev']:.4f}")
    print(f"{'En Düşük Frekans (Nadir)':<30} | {m_pi['nadir']:.4f}         | {m_ai['nadir']:.4f}")
    print(f"{'Kontrol Eforu (Total MW)':<30} | {toplam_mw_pi:.2f} MW      | {toplam_mw_ai:.2f} MW")
    print(f"{'Sinyal Titremesi (Jitter)':<30} | {m_pi['jitter']:.2f}          | {m_ai['jitter']:.2f}")
    print(f"{'Otonomi Oranı (%)':<30} | %0.00           | %{m_ai['autonomy']:.2f}")
    print("-" * 65)
    print(f"🚀 NET ENERJİ TASARRUFU      : %{tasarruf_yuzdesi:.2f}")
    print(f"🌍 ENGELLENEN CO2 SALINIMI   : {engellenen_co2:.2f} kg")
    print("="*65 + "\n")

    # Grafikleme
    plt.style.use('default') 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    
    ax1.plot(delta_f_pi, label="Geleneksel PI", color='#d62728', linestyle='--', alpha=0.6)
    ax1.plot(delta_f_hybrid, label="Önerilen Safe-AI", color='#1f77b4', linewidth=2)
    ax1.axhline(y=0.15, color='black', linestyle=':', alpha=0.3, label='Güvenlik Sınırı')
    ax1.axhline(y=-0.15, color='black', linestyle=':', alpha=0.3)
    ax1.set_ylabel("Frekans Sapması (Hz)")
    ax1.set_title("Sistem Kararlılık Analizi")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    ax2.plot(actions_pi, color='#d62728', alpha=0.3, linestyle='--')
    ax2.plot(actions_hybrid, color='#1f77b4', alpha=0.8)
    ax2.set_ylabel("Batarya Gücü (pu)")
    ax2.set_title(f"Kontrol Eforu (Tasarruf: %{tasarruf_yuzdesi:.2f})")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('Kapsamli_Performans_Analizi.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_evaluation()