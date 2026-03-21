"""
evaluate.py
===========
Orijinal evaluate.py'nin düzeltilmiş versiyonu.
Import yolları projeyle birebir aynı.

Düzeltilen hatalar:
  1. obs[0] Hz hatası: delta_f_hybrid.append(obs[0]) → info["delta_f"]
  2. Adil karşılaştırma: PI +0.4 → +0.2 (Safe-AI ile aynı yük darbesi)
  3. wind_profile: reset() SONRASINDA atandı
  4. CO2 birimi: Σ|p|*dt*P_base/3600 → [MWh], sonra *400 kg/MWh
  5. SafeHybridController.reset() çağrısı eklendi
  6. PI baseline: obs_pi[0] * 0.3 ile Hz'e çevriliyor
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs.microgrid_lfc_env import MicrogridLFCEnv
from agents.safe_hybrid_controller import SafeHybridController
from utils.Data_loader import load_epias_wind_data


def calculate_metrics(delta_f_hz, p_ess_real, safety_flags,
                       dt=0.1, base_power_mw=10.0):
    """
    Akademik raporlama için detaylı metrik hesaplamaları.

    delta_f_hz : info["delta_f"] — gerçek Hz, obs[0] DEĞİL
    p_ess_real : info["p_ess"]   — slew sonrası gerçek değer
    """
    delta_f = np.array(delta_f_hz)
    actions  = np.array(p_ess_real)
    N        = len(delta_f)

    # Kararlılık
    rmse    = float(np.sqrt(np.mean(delta_f ** 2)))
    nadir   = float(np.min(delta_f))
    zenith  = float(np.max(delta_f))
    max_dev = float(np.max(np.abs(delta_f)))

    # Enerji (doğru birim zinciri)
    effort_pu_s = float(np.sum(np.abs(actions)) * dt)
    effort_mwh  = effort_pu_s * base_power_mw / 3600.0

    # Jitter — gerçek p_ess farkları
    jitter = float(np.sum(np.abs(np.diff(actions))))

    # Otonomi
    ai_active_steps = int(np.sum(safety_flags))
    autonomy_ratio  = (ai_active_steps / max(N, 1)) * 100.0

    return {
        "rmse":        rmse,
        "nadir":       nadir,
        "zenith":      zenith,
        "max_dev":     max_dev,
        "effort_mwh":  effort_mwh,
        "jitter":      jitter,
        "autonomy":    autonomy_ratio,
    }


def run_evaluation():
    print("Test ortamı ve veriler hazırlanıyor...")

    csv_path = "data/Gercek_Zamanli_Uretim-20112025-20022026.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/epias_wind_data.csv"

    # ── Ortamlar ──────────────────────────────────────────────────
    env    = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
    env_pi = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)

    # Görülmemiş test verisi
    full_data        = load_epias_wind_data(filepath=csv_path, max_steps=2000)
    unseen_test_data = full_data[1000:2000].astype(np.float32)

    # DÜZELTME: reset() SONRASINDA wind_profile ata
    obs,    _ = env.reset(seed=2024)
    obs_pi, _ = env_pi.reset(seed=2024)
    env.wind_profile    = unseen_test_data
    env_pi.wind_profile = unseen_test_data


    
    # ── Model ─────────────────────────────────────────────────────
    model_path = "models/td3_microgrid_model"

    # Python sürüm farkından kaynaklanan deserialization hatasını engellemek için:
    custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0  # Eğer PPO kullanılsaydı gerekirdi, TD3 için de eklenebilir.
}
    try:
        model = TD3.load(model_path, custom_objects=custom_objects)
    except Exception:
        model = TD3.load(model_path + ".zip", custom_objects=custom_objects)

    # ── Hibrit Kontrolcü (DÜZELTME: reset() çağrıldı) ─────────────
    hybrid_controller = SafeHybridController(
        ai_model     = model,
        Kp           = 2.0,
        Ki           = 0.5,
        safety_limit = 0.15
    )
    hybrid_controller.reset()

    delta_f_hybrid, p_ess_hybrid, safety_flags = [], [], []
    delta_f_pi,     p_ess_pi                   = [], []

    # ── Senaryo 1: Safe-AI ────────────────────────────────────────
    print("Simülasyon 1: Safe-AI çalışıyor...")
    for step in range(1000):
        if step == 400:
            env.base_load += 0.2

        action, active_mode = hybrid_controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # DÜZELTME: info'dan gerçek fiziksel değerleri al
        delta_f_hybrid.append(info["delta_f"])
        p_ess_hybrid.append(info["p_ess"])
        safety_flags.append(1 if active_mode == "AI" else 0)

        if terminated or truncated:
            break

    # ── Senaryo 2: PI Baseline ────────────────────────────────────
    print("Simülasyon 2: PI çalışıyor...")
    integral_error_pi = 0.0
    for step in range(1000):
        if step == 400:
            env_pi.base_load += 0.2   # DÜZELTME: +0.4 → +0.2

        # DÜZELTME: obs_pi[0] normalleştirilmiş → Hz'e çevir
        delta_f_hz         = float(obs_pi[0]) * 0.3
        error              = 0.0 - delta_f_hz
        integral_error_pi += error * 0.1
        integral_error_pi  = np.clip(integral_error_pi, -0.5, 0.5)
        action_pi          = float(np.clip(
            (2.0 * error) + (0.5 * integral_error_pi), -1.0, 1.0
        ))

        obs_pi, _, term, trunc, info_pi = env_pi.step([action_pi])
        delta_f_pi.append(info_pi["delta_f"])
        p_ess_pi.append(info_pi["p_ess"])

        if term or trunc:
            break

    # ── Hesaplamalar ──────────────────────────────────────────────
    BASE_POWER_MW = 10.0
    CO2_FACTOR    = 400.0  # kg CO2 / MWh

    m_ai = calculate_metrics(delta_f_hybrid, p_ess_hybrid, safety_flags,
                              base_power_mw=BASE_POWER_MW)
    m_pi = calculate_metrics(delta_f_pi, p_ess_pi, [0]*len(delta_f_pi),
                              base_power_mw=BASE_POWER_MW)

    tasarruf_mwh     = m_pi["effort_mwh"] - m_ai["effort_mwh"]
    tasarruf_yuzdesi = (tasarruf_mwh / max(m_pi["effort_mwh"], 1e-9)) * 100.0
    engellenen_co2   = tasarruf_mwh * CO2_FACTOR

    # ── Rapor ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 KAPSAMLI MÜHENDİSLİK VE PERFORMANS ANALİZİ 📊")
    print("=" * 65)
    print(f"{'Metrik':<30} | {'Geleneksel PI':<15} | {'Önerilen Safe-AI':<15}")
    print("-" * 65)
    print(f"{'Frekans RMSE (mHz)':<30} | {m_pi['rmse']*1000:<15.2f} | {m_ai['rmse']*1000:<15.2f}")
    print(f"{'Maksimum Sapma (mHz)':<30} | {m_pi['max_dev']*1000:<15.2f} | {m_ai['max_dev']*1000:<15.2f}")
    print(f"{'En Düşük Frekans (mHz)':<30} | {m_pi['nadir']*1000:<15.2f} | {m_ai['nadir']*1000:<15.2f}")
    print(f"{'Kontrol Eforu (MWh)':<30} | {m_pi['effort_mwh']:<15.5f} | {m_ai['effort_mwh']:<15.5f}")
    print(f"{'Sinyal Titremesi (Jitter)':<30} | {m_pi['jitter']:<15.2f} | {m_ai['jitter']:<15.2f}")
    print(f"{'Otonomi Oranı (%)':<30} | {'%0.00':<15} | %{m_ai['autonomy']:.2f}")
    print("-" * 65)
    print(f"🚀 NET ENERJİ TASARRUFU      : %{tasarruf_yuzdesi:.2f}  ({tasarruf_mwh:.5f} MWh/ep)")
    print(f"🌍 ENGELLENEN CO2 SALINIMI   : {engellenen_co2:.4f} kg")
    print(f"🛡  AI / PI adım              : "
          f"{hybrid_controller.usage_log['AI']} / {hybrid_controller.usage_log['PI']}")
    print("=" * 65 + "\n")

    # ── Grafik ────────────────────────────────────────────────────
    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    t    = np.arange(len(delta_f_hybrid)) * 0.1
    t_pi = np.arange(len(delta_f_pi))     * 0.1

    ax1.plot(t_pi, delta_f_pi,    label="Geleneksel PI",   color="#d62728",
             linestyle="--", alpha=0.6)
    ax1.plot(t,   delta_f_hybrid, label="Önerilen Safe-AI", color="#1f77b4",
             linewidth=2)
    ax1.axhline(y= 0.15, color="black", linestyle=":", alpha=0.3,
                label="Güvenlik Sınırı")
    ax1.axhline(y=-0.15, color="black", linestyle=":", alpha=0.3)
    ax1.axvline(x=400*0.1, color="orange", linestyle="-.", alpha=0.6,
                label="Yük Darbesi (+0.2 pu)")
    ax1.set_ylabel("Frekans Sapması (Hz)")
    ax1.set_title("Sistem Kararlılık Analizi")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)

    ax2.plot(t_pi, p_ess_pi,    color="#d62728", alpha=0.4,
             linestyle="--", label="PI")
    ax2.plot(t,   p_ess_hybrid, color="#1f77b4", alpha=0.85,
             label="Safe-AI")
    ax2.set_ylabel("Batarya Gücü (pu)")
    ax2.set_xlabel("Zaman (s)")
    ax2.set_title(f"Kontrol Eforu  |  Tasarruf: %{tasarruf_yuzdesi:.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("Kapsamli_Performans_Analizi.png", dpi=300)
    print("[GÖRSEL] Kapsamli_Performans_Analizi.png kaydedildi.")
    plt.show()


if __name__ == "__main__":
    run_evaluation()