import numpy as np

class SafeHybridController:
    def __init__(self, ai_model, Kp=2.0, Ki=0.5, safety_limit=0.15):
        self.ai_model = ai_model
        self.Kp = Kp
        self.Ki = Ki
        self.limit = safety_limit
        
        self.integral_error = 0.0
        self.usage_log = {"AI": 0, "PI": 0}

    def predict(self, obs, deterministic=True):
        delta_f = obs[0]
        
        # Güvenlik Kalkanı Mantığı
        if abs(delta_f) > self.limit:
            # TEHLİKE: PI Kontrolcü Devreye Girer
            self.usage_log["PI"] += 1
            error = 0.0 - delta_f
            self.integral_error += error * 0.1 # dt = 0.1 varsayımı
            
            action = (self.Kp * error) + (self.Ki * self.integral_error)
            action = np.clip(action, -1.0, 1.0)
            return np.array([action]), "PI"
        else:
            # GÜVENLİ: DDPG Yapay Zeka Devrede
            self.usage_log["AI"] += 1
            self.integral_error = 0.0 # Geçiş sıçramalarını önlemek için sıfırla
            action, _ = self.ai_model.predict(obs, deterministic=deterministic)
            return action, "AI"