"""
config.py — Paylaşılan sözleşme (tek doğruluk kaynağı)
=======================================================
Tüm fiziksel parametreler, ölçek sabitleri ve veri tipleri burada tanımlıdır.
Başka hiçbir modülde sihirli sayı (0.2, 0.3, 0.15 ...) bulunmamalıdır.

Birimler
--------
- Güç        : sistem tabanında per-unit (S_base = GridParams.s_base_mw)
- Frekans    : dinamik model içinde per-unit (Δf_pu = Δf_Hz / f0); dış arayüzde Hz
- Zaman      : saniye
- İşaret     : P_ess > 0 → batarya DEŞARJ (şebekeye güç verir)
               ΔP_load > 0 → yük artışı (frekansı düşürür)
               ΔP_wind > 0 → rüzgâr üretimi artışı (frekansı yükseltir)

Ayrıntılı tanımlar için: docs/SPEC.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "epias_wind_data.csv"
WIND_COLUMN = "Rüzgar"


# ══════════════════════════════════════════════════════════════════
#  Fiziksel parametreler
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GridParams:
    """İzole mikro şebeke: dizel jeneratör (governor + türbin + yavaş AGC)."""
    s_base_mw: float = 10.0   # sistem taban gücü [MW]
    f0_hz: float = 50.0       # nominal frekans [Hz]
    H: float = 2.0            # atalet sabiti [s]  (M = 2H)
    D: float = 1.0            # yük sönümü [pu güç / pu frekans]
    R: float = 0.05           # governor droop [pu frekans / pu güç]
    T_g: float = 0.1          # governor zaman sabiti [s]
    T_t: float = 0.4          # türbin zaman sabiti [s]
    K_i_agc: float = 0.1      # dizel AGC integral kazancı [1/s]
    dt: float = 0.1           # kontrol adımı [s]

    @property
    def beta(self) -> float:
        """Frekans öngerilim katsayısı β = D + 1/R [pu güç / pu frekans]."""
        return self.D + 1.0 / self.R


@dataclass(frozen=True)
class BessParams:
    """Batarya enerji depolama sistemi."""
    p_max: float = 0.2          # güç sınırı [pu, sistem tabanı]
    ramp_rate: float = 2.0      # komut hız sınırı [pu/s]  (0,4'te hız sınırı kaynaklı limit çevrim oluşuyor)
    T_ess: float = 0.1          # dönüştürücü zaman sabiti [s]
    e_mwh: float = 0.5          # enerji kapasitesi [MWh]
    eta: float = 0.95           # tek yön verim (şarj ve deşarj ayrı ayrı)
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_ref: float = 0.5        # hedef SOC


@dataclass(frozen=True)
class ShieldParams:
    """PI güvenlik kalkanı (histerezisli)."""
    engage_hz: float = 0.20     # |Δf| bu değeri AŞINCA kalkan devreye girer
                                # (ENTSO-E azami kalıcı sapma; varsayılan PI/PID'in en kötü değerinin üstünde,
                                #  bkz. docs/MODEL_VALIDATION.md §4)
    release_hz: float = 0.10    # |Δf| bu değerin ALTINA inince kalkan bırakır
    kp: float = 5.0             # [normalize aksiyon / Hz]
    ki: float = 2.0             # [normalize aksiyon / (Hz·s)]
    min_hold_s: float = 1.0     # kalkan devreye girdikten sonra en az bu kadar aktif kalır [s]
    penalty: float = 0.5        # kalkan aktifken adım başına ödül cezası


@dataclass(frozen=True)
class RewardWeights:
    """Adım ödülü = −(w_f·ef² + w_u·eu² + w_j·ej² + w_soc·es²). Bkz. SPEC §6."""
    w_f: float = 1.0            # frekans sapması
    w_u: float = 0.1            # batarya gücü (yıpranma / enerji)
    w_j: float = 0.5            # komut değişimi (jitter)
    w_soc: float = 0.0          # SOC sapması (varsayılan kapalı: AGC bataryayı zaten boşaltır, bkz. SPEC §6)
    f_ref_hz: float = 0.1       # frekans hatasının normalize edildiği referans


@dataclass(frozen=True)
class ObsScales:
    """Gözlem normalizasyonu. Her bileşen ölçeklendikten sonra [-1, 1]'e kırpılır."""
    f_hz: float = 0.25          # Δf [Hz]
    rocof_hz_s: float = 1.0     # dΔf/dt [Hz/s]
    int_hz_s: float = 1.0       # ∫Δf dt [Hz·s]  (integral de bu değerde sınırlanır)


@dataclass(frozen=True)
class ScenarioParams:
    """Bozucu senaryo üretimi."""
    episode_s: float = 150.0        # bölüm süresi [s]
    wind_share: float = 0.4         # rüzgâr kurulu gücü [pu, sistem tabanı]
    turb_intensity: float = 0.04    # OU türbülansının durağan std'si (ortalama seviyeye oranla)
    turb_tau_s: float = 5.0         # OU zaman sabiti [s]
    ramp_prob: float = 0.3          # bölümde rüzgâr rampası olma olasılığı
    ramp_size: tuple[float, float] = (0.05, 0.15)     # rampa büyüklüğü [wind_share oranı]
    ramp_duration_s: tuple[float, float] = (20.0, 40.0)
    load_noise_std: float = 0.002   # yük gürültüsü [pu]
    n_load_steps: tuple[int, int] = (1, 3)            # bölüm başına yük adımı sayısı (dahil)
    load_step_size: tuple[float, float] = (0.02, 0.10)  # |adım| [pu], işaret rastgele
    load_step_window_s: tuple[float, float] = (10.0, 120.0)  # adımların düşebileceği zaman aralığı
    load_step_min_gap_s: float = 15.0
    soc0_range: tuple[float, float] = (0.3, 0.7)
    train_frac: float = 0.7         # kronolojik eğitim/test bölmesi

    def n_steps(self, dt: float) -> int:
        return int(round(self.episode_s / dt))


@dataclass(frozen=True)
class BaselineGains:
    """Referans kontrolcülerin varsayılan kazançları [normalize aksiyon / Hz ...].

    tune_baselines.py bunların yerine results/baseline_gains.json üretir;
    o dosya varsa evaluate.py onu kullanır.
    """
    droop_k: float = 5.0
    pi_kp: float = 5.0
    pi_ki: float = 2.0
    pid_kp: float = 5.0
    pid_ki: float = 2.0
    pid_kd: float = 0.5


@dataclass(frozen=True)
class EvalParams:
    n_scenarios: int = 40
    seed: int = 20240
    settle_band_hz: float = 0.02
    bootstrap_samples: int = 2000


# ══════════════════════════════════════════════════════════════════
#  Veri tipleri
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Scenario:
    """Bir bölümün tüm dış bozucuları. Uzunluk = bölümdeki adım sayısı.

    Değerler başlangıç denge noktasından SAPMA'dır; k=0'da sistem dengededir.
    Net bozucu: d_net[k] = d_wind[k] − d_load[k]   [pu]
    """
    d_wind: np.ndarray                 # ΔP_wind [pu], shape (n_steps,)
    d_load: np.ndarray                 # ΔP_load [pu], shape (n_steps,)
    soc0: float = 0.5
    wind_level: float = 0.5            # EPİAŞ normalize seviyesi [0,1] (çalışma noktası)
    meta: dict[str, Any] = field(default_factory=dict)
    # meta anahtarları (üretici doldurur):
    #   "load_steps": list[tuple[int, float]]  → (adım indeksi, büyüklük pu)
    #   "wind_ramp" : tuple[int, int, float] | None → (başlangıç, bitiş, büyüklük pu)
    #   "hour_index": int, "bank_index": int (yalnızca make_scenario_bank)

    @property
    def n_steps(self) -> int:
        return int(len(self.d_wind))

    @property
    def d_net(self) -> np.ndarray:
        return np.asarray(self.d_wind, dtype=float) - np.asarray(self.d_load, dtype=float)


def make_step_scenario(n_steps: int, step_at: int, size: float, soc0: float = 0.5) -> Scenario:
    """Test ve doğrulama için: tek bir yük adımı, gürültüsüz."""
    d_load = np.zeros(n_steps, dtype=float)
    d_load[step_at:] = size
    return Scenario(
        d_wind=np.zeros(n_steps, dtype=float),
        d_load=d_load,
        soc0=soc0,
        meta={"load_steps": [(int(step_at), float(size))], "wind_ramp": None},
    )


# Kontrolcü adları (results.json anahtarları) — değiştirmeyin
CTRL_NOBESS = "NoBESS"
CTRL_DROOP = "Droop"
CTRL_PI = "PI"
CTRL_PID = "PID"
CTRL_TD3_SHIELD = "TD3+Shield"
CTRL_SAC_SHIELD = "SAC+Shield"
CTRL_TD3_RAW = "TD3"
ALL_CONTROLLERS = (
    CTRL_NOBESS, CTRL_DROOP, CTRL_PI, CTRL_PID,
    CTRL_TD3_SHIELD, CTRL_SAC_SHIELD, CTRL_TD3_RAW,
)
