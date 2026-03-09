"""
agents/safe_hybrid_controller.py
=================================
Orijinal kodunun düzeltilmiş versiyonu.
Sınıf adı, metot imzaları, usage_log yapısı — hepsi aynı.

Düzeltilen hatalar:
  1. obs[0] ölçek hatası: delta_f = obs[0] → delta_f_hz = obs[0] * 0.3
     Önceki hali: 0.15 eşiği → 0.045 Hz'de devreye giriyordu (3.3x erken)
  2. Bumpless transfer: integral = 0.0 (ani) → integral *= 0.95 (yumuşak)
  3. reset() metodu eklendi (çok-episode testler için zorunlu)
  4. Windup koruması eklendi: integral clip ±0.5
"""

import numpy as np


class SafeHybridController:
    def __init__(self, ai_model, Kp=2.0, Ki=0.5, safety_limit=0.15):
        self.ai_model = ai_model
        self.Kp = Kp
        self.Ki = Ki
        self.limit = safety_limit        # Hz cinsinden fiziksel eşik
        self.obs_scale = 0.3             # obs[0] → Hz dönüşüm katsayısı

        self.integral_error = 0.0
        self.usage_log = {"AI": 0, "PI": 0}

    def reset(self):
        """Her episode başında çağır — usage_log ve integral sıfırlanır."""
        self.integral_error = 0.0
        self.usage_log = {"AI": 0, "PI": 0}

    def predict(self, obs, deterministic=True):
        # DÜZELTME 1: obs[0] normalleştirilmiş (÷0.3) — Hz'e çevir
        # Orijinal:   delta_f = obs[0]           → eşik 0.045 Hz'deydi!
        # Düzeltilmiş: delta_f_hz = obs[0] * 0.3 → eşik doğru: 0.15 Hz
        delta_f_hz = float(obs[0]) * self.obs_scale

        if abs(delta_f_hz) > self.limit:
            # TEHLİKE: PI Kontrolcü Devreye Girer
            self.usage_log["PI"] += 1
            error = 0.0 - delta_f_hz
            self.integral_error += error * 0.1  # dt = 0.1

            # DÜZELTME 4: Windup koruması (orijinalde yoktu)
            self.integral_error = np.clip(self.integral_error, -0.5, 0.5)

            action = (self.Kp * error) + (self.Ki * self.integral_error)
            action = float(np.clip(action, -1.0, 1.0))
            return np.array([action], dtype=np.float32), "PI"

        else:
            # GÜVENLİ: AI Devrede
            self.usage_log["AI"] += 1

            # DÜZELTME 2: Bumpless transfer — ani sıfır yerine yavaş decay
            # Orijinal:    self.integral_error = 0.0   → geçişte eylem sıçrar
            # Düzeltilmiş: self.integral_error *= 0.95 → yumuşak geçiş
            self.integral_error *= 0.95

            action, _ = self.ai_model.predict(obs, deterministic=deterministic)
            return action, "AI"

    @property
    def autonomy_ratio(self):
        """Episode boyunca AI aktif adım yüzdesi (0-100)."""
        total = self.usage_log["AI"] + self.usage_log["PI"]
        return (self.usage_log["AI"] / max(total, 1)) * 100.0