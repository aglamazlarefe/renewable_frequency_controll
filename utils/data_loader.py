import pandas as pd
import numpy as np

def load_epias_wind_data(filepath="data/epias_wind_data.csv", max_steps=1000):
    """
    EPİAŞ CSV verisini okur, Türkçe sayı formatını düzeltir ve normalize eder.
    """
    try:
        # EPİAŞ verilerinde ayraç noktalı virgül (;), ondalık virgül (,), binlik ayracı nokta (.)
        df = pd.read_csv(filepath, sep=';', decimal=',', thousands='.')
        
        # Sütun adı yüklenen CSV dosyası ile birebir eşleştirildi
        wind_column = 'Rüzgar' 
        
        # Sadece rüzgar verisini al ve boşlukları (NaN) temizle
        wind_power = df[wind_column].dropna().values
        
        # Veriyi Normalize Et (0 ile 1 arasına çek)
        max_power = np.max(wind_power)
        
        # 0'a bölme hatasını engellemek için güvenlik kontrolü (DRL algoritmalarında nan değerleri engeller)
        if max_power == 0:
            return np.zeros(max_steps)
            
        normalized_wind = wind_power / max_power
        
        # Simülasyon adımlarına göre interpolasyon uygula
        x_original = np.linspace(0, 1, len(normalized_wind))
        x_target = np.linspace(0, 1, max_steps)
        interpolated_wind = np.interp(x_target, x_original, normalized_wind)
        
        return interpolated_wind
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("CSV dosyasının yolunu ve sütun adını kontrol et.")
        # Hata anında sistemin çökmesini engellemek için sıfır dizisi döndür
        return np.zeros(max_steps)