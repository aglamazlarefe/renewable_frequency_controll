import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from envs.microgrid_lfc_env import MicrogridLFCEnv
from agents.safe_hybrid_controller import SafeHybridController

# 1. Ortamı ve Modeli Yükle
print("Test ortamı yükleniyor...")
env = MicrogridLFCEnv(data_path="data/epias_wind_data.csv", max_steps=1000)
# Kendi eğittiğin modeli yükle
drl_model = DDPG.load("models/ddpg_microgrid_model")

# 2. Güvenli Hibrit Kontrolcüyü (Safe AI) Başlat
# Güvenlik sınırı 0.15 Hz olarak belirlendi
hybrid_controller = SafeHybridController(ai_model=drl_model, Kp=2.0, Ki=0.5, safety_limit=0.15)

# 3. Klasik PI Kontrolcü için Ayrı Bir Ortam (Kıyaslama için)
env_pi = MicrogridLFCEnv(data_path="data/epias_wind_data.csv", max_steps=1000)

# Veri Saklama Listeleri
delta_f_hybrid = []
delta_f_pi = []
actions_hybrid = []
actions_pi = []
safety_flags = [] # AI mı devrede (1), PI mı (0)?

# --- HİBRİT (SAFE AI) SİMÜLASYONU ---
obs, _ = env.reset()
for _ in range(1000):
    action, active_mode = hybrid_controller.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    delta_f_hybrid.append(obs[0])
    actions_hybrid.append(action[0])
    safety_flags.append(1 if active_mode == "AI" else 0)
    
    if terminated or truncated:
        break

# --- SADECE PI KONTROLCÜ SİMÜLASYONU ---
obs_pi, _ = env_pi.reset()
integral_error_pi = 0.0
for _ in range(1000):
    error = 0.0 - obs_pi[0]
    integral_error_pi += error * 0.1
    action_pi = (2.0 * error) + (0.5 * integral_error_pi)
    action_pi = np.clip(action_pi, -1.0, 1.0)
    
    obs_pi, _, term, trunc, _ = env_pi.step([action_pi])
    
    delta_f_pi.append(obs_pi[0])
    actions_pi.append(action_pi)
    
    if term or trunc:
        break

# 4. CO2 VE TASARRUF HESAPLAMALARI
EMISYON_FAKTORU = 0.4 # Doğalgaz emisyon faktörü (kg CO2 / kWh varsayımı)
# Simülasyon gücünü (pu) sanal bir MW/h değerine ölçekle (Örn: 10 MW baz güç)
BASE_POWER_MW = 10 

toplam_enerji_pi = np.sum(np.abs(actions_pi)) * BASE_POWER_MW
toplam_enerji_hybrid = np.sum(np.abs(actions_hybrid)) * BASE_POWER_MW

tasarruf_mwh = (toplam_enerji_pi - toplam_enerji_hybrid) / 1000 # Kabaca birim çevrimi
engellenen_co2 = tasarruf_mwh * EMISYON_FAKTORU * 1000 # kg cinsinden

print("-" * 30)
print("İKLİM ETKİSİ SONUÇLARI:")
print(f"PI Kontrol Eforu: {toplam_enerji_pi:.2f} birim")
print(f"Safe AI Eforu: {toplam_enerji_hybrid:.2f} birim")
print(f"Net Engellenen CO2 Salınımı: {engellenen_co2:.2f} kg")
print("-" * 30)

# 5. GRAFİK ÇİZİMLERİ (Rapor İçin)
plt.figure(figsize=(12, 6))

# Üst Grafik: Frekans Sapması
plt.subplot(2, 1, 1)
plt.plot(delta_f_pi, label="Klasik PI Kontrol", color='red', linestyle='dashed')
plt.plot(delta_f_hybrid, label="Safe AI (DDPG+PI)", color='blue')
plt.axhline(y=0.15, color='gray', linestyle='dotted', label='Güvenlik Sınırı')
plt.axhline(y=-0.15, color='gray', linestyle='dotted')
plt.title("EPİAŞ Gerçek Rüzgar Verisi Altında Frekans Kararlılığı (Safe AI vs PI)")
plt.ylabel("Δf (Hz)")
plt.legend()
plt.grid(True)

# Alt Grafik: Hangi Kontrolcü Devrede (Güvenlik Kalkanı Aksiyonu)
plt.subplot(2, 1, 2)
plt.plot(safety_flags, label="1 = AI Devrede, 0 = PI (Acil Durum)", color='green')
plt.title("Güvenlik Kalkanı (Safety Shield) Durumu")
plt.xlabel("Zaman Adımı")
plt.ylabel("Aktif Kontrolcü")
plt.yticks([0, 1], ['PI (Fail-Safe)', 'AI'])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()