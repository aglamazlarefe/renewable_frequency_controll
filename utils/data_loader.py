"""EPİAŞ rüzgâr verisi yükleyici (SPEC §3.1).

Saatlik rüzgâr üretimini okur ve en büyük değere bölerek [0, 1] aralığına getirir.
Hatalar yutulmaz: dosya yoksa FileNotFoundError, sütun yoksa veya veri boşsa ValueError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_DATA_PATH, WIND_COLUMN


def load_wind_levels(path: str | Path = DEFAULT_DATA_PATH) -> np.ndarray:
    """Saatlik rüzgâr seviyelerini döndürür (float64, en büyük değere bölünmüş)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Rüzgâr veri dosyası bulunamadı: {path}")

    # EPİAŞ biçimi: ayraç ';', ondalık ',', binlik '.'
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".")
    if WIND_COLUMN not in df.columns:
        raise ValueError(f"'{WIND_COLUMN}' sütunu yok; mevcut sütunlar: {list(df.columns)}")

    values = pd.to_numeric(df[WIND_COLUMN], errors="raise").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"'{WIND_COLUMN}' sütununda geçerli veri yok: {path}")

    peak = float(values.max())
    if peak <= 0.0:
        raise ValueError("Rüzgâr verisinin en büyük değeri pozitif değil; normalize edilemez.")
    return values / peak


def split_wind_levels(levels: np.ndarray, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Kronolojik bölme: ilk int(train_frac·n) eğitim, kalanı test."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac (0, 1) aralığında olmalı: {train_frac}")
    levels = np.asarray(levels)
    n_train = int(train_frac * len(levels))
    return levels[:n_train], levels[n_train:]
