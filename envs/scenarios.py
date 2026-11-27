"""Bozucu senaryo üretimi (SPEC §3.3 ve §5).

Rüzgâr çalışma noktası EPİAŞ saatlik verisinden, saniye ölçekli değişim sentetik
OU türbülansı + isteğe bağlı rampa + yük adımlarıyla üretilir.
Tüm rastgelelik yalnızca verilen `np.random.Generator` üzerinden gelir.
"""

from __future__ import annotations

import numpy as np

from config import DEFAULT_DATA_PATH, GridParams, Scenario, ScenarioParams
from utils.data_loader import load_wind_levels, split_wind_levels


class ScenarioGenerator:
    """Verilen rüzgâr seviyelerinden bölüm senaryoları üretir."""

    def __init__(self, wind_levels: np.ndarray,
                 params: ScenarioParams = ScenarioParams(),
                 grid: GridParams = GridParams()):
        wind_levels = np.asarray(wind_levels, dtype=np.float64)
        if wind_levels.ndim != 1 or wind_levels.size == 0:
            raise ValueError("wind_levels boş olmayan 1B bir dizi olmalı.")
        self.wind_levels = wind_levels
        self.params = params
        self.grid = grid

    # ── parçalar ──────────────────────────────────────────────────
    def _turbulence(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Kesin ayrıklaştırılmış OU süreci; durağan std = turb_intensity, x[0] = 0."""
        p, dt = self.params, self.grid.dt
        decay = np.exp(-dt / p.turb_tau_s)
        noise_gain = p.turb_intensity * np.sqrt(1.0 - np.exp(-2.0 * dt / p.turb_tau_s))
        z = rng.standard_normal(n)
        x = np.zeros(n)
        for k in range(n - 1):
            x[k + 1] = x[k] * decay + noise_gain * z[k + 1]
        return x

    def _ramp(self, rng: np.random.Generator, n: int):
        """Rampa olursa (başlangıç, bitiş, büyüklük) ve d_wind'e eklenecek eğri döner."""
        p, dt = self.params, self.grid.dt
        if rng.random() >= p.ramp_prob:
            return None, np.zeros(n)
        size = rng.uniform(*p.ramp_size) * p.wind_share * rng.choice((-1.0, 1.0))
        dur = max(1, int(round(rng.uniform(*p.ramp_duration_s) / dt)))
        dur = min(dur, n - 1)
        start = int(rng.integers(0, n - dur))
        k = np.arange(n)
        curve = size * np.clip((k - start) / dur, 0.0, 1.0)
        return (start, start + dur, float(size)), curve

    def _load_step_indices(self, rng: np.random.Generator, count: int) -> list[int]:
        """Pencere ve minimum aralık kurallarına uyan, sıralı adım indeksleri."""
        p, dt = self.params, self.grid.dt
        lo = int(np.ceil(p.load_step_window_s[0] / dt - 1e-9))
        hi = int(np.floor(p.load_step_window_s[1] / dt + 1e-9))
        gap = int(np.ceil(p.load_step_min_gap_s / dt - 1e-9))
        span = hi - lo - (count - 1) * gap
        if span < 0:
            raise ValueError("Yük adımları pencereye minimum aralıkla sığmıyor.")
        base = np.sort(rng.integers(0, span + 1, size=count))
        return [int(lo + b + i * gap) for i, b in enumerate(base)]

    # ── ana giriş ─────────────────────────────────────────────────
    def sample(self, rng: np.random.Generator) -> Scenario:
        p = self.params
        n = p.n_steps(self.grid.dt)

        hour = int(rng.integers(0, len(self.wind_levels)))
        level = float(self.wind_levels[hour])

        d_wind = p.wind_share * level * self._turbulence(rng, n)
        ramp, ramp_curve = self._ramp(rng, n)
        d_wind = d_wind + ramp_curve
        d_wind[0] = 0.0

        count = int(rng.integers(p.n_load_steps[0], p.n_load_steps[1] + 1))
        idx = self._load_step_indices(rng, count)
        d_load = np.zeros(n)
        steps: list[tuple[int, float]] = []
        for k in idx:
            size = float(rng.uniform(*p.load_step_size) * rng.choice((-1.0, 1.0)))
            d_load[k:] += size
            steps.append((int(k), size))
        d_load = d_load + rng.normal(0.0, p.load_noise_std, size=n)
        d_load[0] = 0.0

        soc0 = float(rng.uniform(*p.soc0_range))
        return Scenario(
            d_wind=d_wind, d_load=d_load, soc0=soc0, wind_level=level,
            meta={"load_steps": sorted(steps), "wind_ramp": ramp, "hour_index": hour},
        )


def make_scenario_bank(wind_levels: np.ndarray, n: int, seed: int,
                       params: ScenarioParams = ScenarioParams(),
                       grid: GridParams = GridParams()) -> list[Scenario]:
    """Aynı seed her zaman aynı bankayı üretir; her senaryo bağımsız çocuk üreteç kullanır."""
    gen = ScenarioGenerator(wind_levels, params, grid)
    children = np.random.SeedSequence(seed).spawn(n)
    bank = []
    for i, child in enumerate(children):
        sc = gen.sample(np.random.default_rng(child))
        sc.meta["bank_index"] = i
        bank.append(sc)
    return bank


def default_generator(split: str = "train", data_path=DEFAULT_DATA_PATH,
                      params: ScenarioParams = ScenarioParams(),
                      grid: GridParams = GridParams()) -> ScenarioGenerator:
    """EPİAŞ verisinin kronolojik 'train' veya 'test' bölmesinden üreteç kurar."""
    if split not in ("train", "test"):
        raise ValueError(f"split 'train' veya 'test' olmalı: {split!r}")
    train, test = split_wind_levels(load_wind_levels(data_path), params.train_frac)
    return ScenarioGenerator(train if split == "train" else test, params, grid)
