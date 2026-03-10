"""
agents/safe_hybrid_controller.py
=================================
Düzeltmeler:
  1. obs[0] ölçek hatası: delta_f_hz = obs[0] * 0.3
  2. Bumpless transfer: integral *= 0.95
  3. reset() metodu eklendi
  4. Windup koruması: integral clip ±0.5
  5. obs_slice parametresi eklendi:
     Eski DDPG modeli 2-boyutlu obs ile eğitildiyse
     obs_slice=2 geçilerek obs[:2] modele gönderilir.
"""

import numpy as np


class SafeHybridController:
    def __init__(self, ai_model, Kp=2.0, Ki=0.5, safety_limit=0.15, obs_slice=None):
        self.ai_model  = ai_model
        self.Kp        = Kp
        self.Ki        = Ki
        self.limit     = safety_limit   # Hz cinsinden fiziksel esik
        self.obs_scale = 0.3            # obs[0] -> Hz donusum katsayisi
        # obs_slice: None = tam obs kullan, int = obs[:obs_slice] kullan
        self.obs_slice = obs_slice

        self.integral_error = 0.0
        self.usage_log = {"AI": 0, "PI": 0}

    def reset(self):
        """Her episode basinda cagir."""
        self.integral_error = 0.0
        self.usage_log = {"AI": 0, "PI": 0}

    def predict(self, obs, deterministic=True):
        delta_f_hz = float(obs[0]) * self.obs_scale

        if abs(delta_f_hz) > self.limit:
            # TEHLIKE: PI Kontrolcu Devreye Girer
            self.usage_log["PI"] += 1
            error = 0.0 - delta_f_hz
            self.integral_error += error * 0.1
            self.integral_error  = np.clip(self.integral_error, -0.5, 0.5)
            action = float(np.clip(
                self.Kp * error + self.Ki * self.integral_error, -1.0, 1.0
            ))
            return np.array([action], dtype=np.float32), "PI"

        else:
            # GUVENLI: AI Devrede
            self.usage_log["AI"] += 1
            self.integral_error *= 0.95  # bumpless transfer

            # obs boyutu uyumsuzlugunu gider
            obs_input = obs[:self.obs_slice] if self.obs_slice is not None else obs
            action, _ = self.ai_model.predict(obs_input, deterministic=deterministic)
            return action, "AI"

    @property
    def autonomy_ratio(self):
        total = self.usage_log["AI"] + self.usage_log["PI"]
        return (self.usage_log["AI"] / max(total, 1)) * 100.0