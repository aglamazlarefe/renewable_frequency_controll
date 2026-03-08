import pandas as pd
import numpy as np

def load_epias_wind_data(filepath="data/epias_wind_data.csv", max_steps=1000):
    """
    EPİAŞ CSV verisini okur, Türkçe sayı formatını düzeltir ve normalize eder.
    """
    try:
        # EPİAŞ verilerinde binlik ayracı nokta, ondalık virgül olabilir.
        df = pd.read_csv(filepath, sep=',', decimal=',', thousands='.')
        
        # Sütun isimleri EPİAŞ'ta farklı olabilir, genellikle 'Rüzgar' veya 'Wind' geçer.
        # İndirdiğin CSV'ye bakıp buradaki sütun adını tam olarak eşleştir!
        wind_column = 'Rüzgar' 
        
        # Sadece rüzgar verisini al ve boşlukları (NaN) temizle
        wind_power = df[wind_column].dropna().values
        
        # Veriyi Normalize Et (0 ile 1 arasına çek)
        max_power = np.max(wind_power)
        normalized_wind = wind_power / max_power
        
        # Elimizdeki veri (örn: 24 saat = 24 veri) simülasyon adımlarından (örn 1000) azsa,
        # aralara interpolasyon (veri uydurma) yaparak saniyelik/milisaniyelik hale getir.
        x_original = np.linspace(0, 1, len(normalized_wind))
        x_target = np.linspace(0, 1, max_steps)
        interpolated_wind = np.interp(x_target, x_original, normalized_wind)
        
        return interpolated_wind
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("CSV dosyasının yolunu ve sütun adını kontrol et.")
        return np.zeros(max_steps) # Hata anında sistem çökmesin diye sıfır döner