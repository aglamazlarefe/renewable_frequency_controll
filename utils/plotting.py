"""Grafikler (SPEC 3.9). Arka uc Agg; ekrana cizim yok.

Tek giris noktasi:
    make_all_figures(example_trajs, example_scenario, delta_f_all, results, out_dir) -> list[Path]

Girdiler
    example_trajs   : kontrolcu adi -> run_episode ciktisi (ayni ornek senaryo)
    example_scenario: Scenario (ilk yuk adimi Fig2'de kullanilir)
    delta_f_all     : kontrolcu adi -> birlestirilmis test Delta f dizisi [Hz]
    results         : SPEC 3.12 sozlugu (controllers / pareto)
Cikti: Fig1..Fig7 sirasiyla 7 PNG yolu.
Eksik kontrolcu, bos Pareto taramasi ve None metrik degerleri desteklenir.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import (  # noqa: E402
    ALL_CONTROLLERS, CTRL_SAC_SHIELD, CTRL_TD3_SHIELD, EvalParams, GridParams,
)

DPI = 200
_KW_PER_MW = 1000.0
_S_PER_H = 3600.0
_MHZ_PER_HZ = 1000.0

FIG_NAMES = (
    "Fig1_Zaman_Serisi.png", "Fig2_Yuk_Adimi_Zoom.png", "Fig3_Frekans_Histogram.png",
    "Fig4_Enerji_SOC.png", "Fig5_Kalkan_Seridi.png", "Fig6_Pareto.png",
    "Fig7_Ozet_Tablo.png",
)

# Kontrolcu basina sabit renk (renk koru dostu) ve cizgi stili
CONTROLLER_COLORS = {
    "NoBESS": "#7f7f7f",
    "Droop": "#e69f00",
    "PI": "#0072b2",
    "PID": "#009e73",
    "TD3+Shield": "#d55e00",
    "SAC+Shield": "#cc79a7",
    "TD3": "#56b4e9",
}
CONTROLLER_STYLES = {
    "NoBESS": ":", "Droop": "--", "PI": "-", "PID": "-.",
    "TD3+Shield": "-", "SAC+Shield": "-", "TD3": "--",
}
_FALLBACK_COLOR = "#444444"

# Ozet tablo satirlari: anahtar -> (Turkce etiket, ondalik basamak)
TABLE_ROWS = (
    ("rmse_mhz", "RMSE [mHz]", 1),
    ("max_dev_mhz", "Maks. sapma [mHz]", 1),
    ("band_100_pct", "±100 mHz bandı [%]", 1),
    ("settling_time_s", "Yerleşme süresi [s]", 1),
    ("bess_throughput_kwh", "BESS enerji geçişi [kWh]", 2),
    ("control_jitter", "Kontrol titreşimi [-]", 1),
    ("autonomy_pct", "Özerklik [%]", 1),
    ("mean_cost", "Ort. maliyet [-]", 3),
)

_SHIELDED = (CTRL_TD3_SHIELD, CTRL_SAC_SHIELD)
_LEGEND_BOTTOM = 0.10      # alt lejant icin ayrilan oran


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "DejaVu Sans",
    })


def _color(name: str) -> str:
    return CONTROLLER_COLORS.get(name, _FALLBACK_COLOR)


def _style(name: str) -> str:
    return CONTROLLER_STYLES.get(name, "-")


def _ordered(names) -> list[str]:
    """ALL_CONTROLLERS sirasi, bilinmeyenler sonda."""
    names = list(names)
    known = [n for n in ALL_CONTROLLERS if n in names]
    return known + [n for n in names if n not in ALL_CONTROLLERS]


def _arr(x: Any) -> np.ndarray:
    """None'lari NaN yaparak float dizisine cevirir."""
    if x is None:
        return np.array([], dtype=float)
    return np.asarray([np.nan if v is None else v for v in np.asarray(x, dtype=object).ravel()],
                      dtype=float)


def _num(v: Any) -> float:
    """None veya sayi olmayan degeri NaN yapar."""
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _metric_stat(results: dict, ctrl: str, key: str, stat: str = "mean") -> float:
    m = (results.get("controllers", {}).get(ctrl, {}).get("metrics", {}) or {}).get(key)
    if isinstance(m, dict):
        return _num(m.get(stat))
    return _num(m)


def _empty_note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _bottom_legend(fig, handles_from_ax, ncol: int = 4) -> None:
    handles, labels = handles_from_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _step_time_s(scenario, dt: float) -> float | None:
    steps = scenario.meta.get("load_steps") if scenario is not None and scenario.meta else None
    if not steps:
        return None
    return min(int(s[0]) for s in steps) * dt


def _time_series(trajs: dict, out: Path, title: str, xlim: tuple[float, float] | None,
                 step_t: float | None, band_hz: float | None) -> Path:
    grid = GridParams()
    names = _ordered(trajs)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
    for n in names:
        tr = trajs[n]
        t = _arr(tr.get("t"))
        axes[0].plot(t, _arr(tr.get("delta_f_hz")), color=_color(n), ls=_style(n), lw=1.4, label=n)
        axes[1].plot(t, _arr(tr.get("p_ess")) * grid.s_base_mw * _KW_PER_MW, color=_color(n),
                     ls=_style(n), lw=1.4, label=n)
    if not names:
        _empty_note(axes[0], "Yörünge verisi yok")
    if band_hz is not None:
        for s in (-band_hz, band_hz):
            axes[0].axhline(s, color="black", lw=0.8, ls=":")
    if step_t is not None:
        for ax in axes:
            ax.axvline(step_t, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("Frekans sapması Δf [Hz]")
    axes[1].set_ylabel("BESS gücü [kW]")
    axes[1].set_xlabel("Zaman [s]")
    axes[0].set_title(title)
    if xlim is not None:
        axes[1].set_xlim(*xlim)
    fig.tight_layout(rect=(0, _LEGEND_BOTTOM, 1, 1))
    _bottom_legend(fig, axes[0])
    return _save(fig, out)


def plot_time_series(trajs: dict, out: Path) -> Path:
    """Fig1: Δf ve BESS gucu zaman serisi."""
    return _time_series(trajs, out, "Örnek senaryoda frekans sapması ve BESS gücü",
                        None, None, None)


def plot_step_zoom(trajs: dict, scenario, out: Path, window_before_s: float = 5.0,
                   window_after_s: float = 40.0) -> Path:
    """Fig2: ilk yuk adimi etrafinda yakinlastirma."""
    dt = GridParams().dt
    step_t = _step_time_s(scenario, dt)
    xlim = None if step_t is None else (max(step_t - window_before_s, 0.0), step_t + window_after_s)
    return _time_series(trajs, out, "İlk yük adımı çevresinde yakınlaştırma", xlim, step_t,
                        EvalParams().settle_band_hz)


def plot_histogram(delta_f_all: dict, out: Path, n_bins: int = 80) -> Path:
    """Fig3: frekans sapmasi histogrami (log eksen)."""
    names = [n for n in _ordered(delta_f_all) if _arr(delta_f_all[n]).size > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if names:
        data = {n: _arr(delta_f_all[n]) for n in names}
        data = {n: v[~np.isnan(v)] for n, v in data.items()}
        pooled = np.concatenate(list(data.values()))
        if pooled.size:
            lim = float(np.max(np.abs(pooled)))
            lim = lim if lim > 0 else 1.0
            bins = np.linspace(-lim, lim, n_bins + 1)
            for n, v in data.items():
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                        color=_color(n), ls=_style(n), label=n)
            ax.set_yscale("log")
        else:
            _empty_note(ax, "Veri yok")
    else:
        _empty_note(ax, "Veri yok")
    ax.set_xlabel("Frekans sapması Δf [Hz]")
    ax.set_ylabel("Olasılık yoğunluğu [1/Hz]")
    ax.set_title("Frekans sapması dağılımı (tüm test senaryoları)")
    fig.tight_layout(rect=(0, _LEGEND_BOTTOM, 1, 1))
    _bottom_legend(fig, ax)
    return _save(fig, out)


def plot_energy_soc(trajs: dict, out: Path) -> Path:
    """Fig4: kumulatif BESS enerji gecisi ve SOC."""
    grid = GridParams()
    names = _ordered(trajs)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
    for n in names:
        tr = trajs[n]
        t = _arr(tr.get("t"))
        kwh = np.cumsum(np.abs(_arr(tr.get("p_ess")))) * grid.s_base_mw * _KW_PER_MW * grid.dt / _S_PER_H
        axes[0].plot(t, kwh, color=_color(n), ls=_style(n), lw=1.4, label=n)
        axes[1].plot(t, _arr(tr.get("soc")), color=_color(n), ls=_style(n), lw=1.4, label=n)
    if not names:
        _empty_note(axes[0], "Yörünge verisi yok")
    axes[0].set_ylabel("Kümülatif enerji geçişi [kWh]")
    axes[1].set_ylabel("Doluluk oranı SOC [-]")
    axes[1].set_xlabel("Zaman [s]")
    axes[0].set_title("BESS enerji geçişi ve SOC")
    fig.tight_layout(rect=(0, _LEGEND_BOTTOM, 1, 1))
    _bottom_legend(fig, axes[0])
    return _save(fig, out)


def plot_shield_strip(trajs: dict, out: Path) -> Path:
    """Fig5: kalkanli RL kontrolculeri icin kalkan etkinlik seridi."""
    names = [n for n in _SHIELDED if n in trajs]
    fig, axes = plt.subplots(max(len(names), 1), 1, figsize=(7.5, 1.6 * max(len(names), 1) + 1.6),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    if not names:
        _empty_note(axes[0], "Kalkanlı RL kontrolcüsü bulunamadı")
    for ax, n in zip(axes, names):
        tr = trajs[n]
        t = _arr(tr.get("t"))
        active = np.asarray(tr.get("shield_active", np.zeros(len(t))), dtype=bool)
        ax.fill_between(t, 0, 1, where=active, step="post", color=_color(n), alpha=0.85, lw=0)
        if t.size:
            ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(False)
        pct = 100.0 * (1.0 - float(np.mean(active))) if active.size else math.nan
        ax.set_ylabel(f"{n}\nözerklik %{pct:.1f}", rotation=0, ha="right", va="center")
    if names:
        axes[-1].set_xlabel("Zaman [s]")
    axes[0].set_title("Kalkan etkinlik şeridi (dolu = kalkan aktif)")
    fig.tight_layout()
    return _save(fig, out)


def plot_pareto(results: dict, out: Path) -> Path:
    """Fig6: Pareto duzlemi (x: enerji gecisi, y: RMSE)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sweep = (results.get("pareto", {}) or {}).get("PI_sweep") or []
    pts = [(_num(p.get("bess_throughput_kwh")), _num(p.get("rmse_mhz"))) for p in sweep]
    pts = sorted(p for p in pts if not (math.isnan(p[0]) or math.isnan(p[1])))
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=_color("PI"), lw=1.2, alpha=0.5, marker="o", ms=3.5,
                label="PI kazanç taraması")
    ctrls = _ordered((results.get("controllers") or {}).keys())
    offsets = ((-12, 14), (12, -18))   # tarama egrisine dik yonlerde, dönüşümlü
    k = 0
    for n in ctrls:
        x = _metric_stat(results, n, "bess_throughput_kwh")
        y = _metric_stat(results, n, "rmse_mhz")
        if math.isnan(x) or math.isnan(y):
            continue
        xe = _metric_stat(results, n, "bess_throughput_kwh", "std")
        ye = _metric_stat(results, n, "rmse_mhz", "std")
        ax.errorbar(x, y, xerr=None if math.isnan(xe) else xe, yerr=None if math.isnan(ye) else ye,
                    fmt="o", ms=8, color=_color(n), capsize=3, elinewidth=1.2, label=n)
        dx, dy = offsets[k % len(offsets)]
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", fontsize=9, color=_color(n))
        k += 1
    if not pts and k == 0:
        _empty_note(ax, "Veri yok")
    ax.set_xlabel("BESS enerji geçişi [kWh]")
    ax.set_ylabel("RMSE [mHz]")
    ax.set_title("Frekans kalitesi – batarya kullanımı uzlaşımı")
    ax.margins(0.12)
    fig.tight_layout(rect=(0, _LEGEND_BOTTOM, 1, 1))
    _bottom_legend(fig, ax, ncol=4)
    return _save(fig, out)


def plot_summary_table(results: dict, out: Path) -> Path:
    """Fig7: ozet metrik tablosu (ortalama ± std)."""
    ctrls = _ordered((results.get("controllers") or {}).keys())
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(ctrls) + 3.2), 4.2))
    ax.axis("off")
    if not ctrls:
        ax.text(0.5, 0.5, "Sonuç verisi yok", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out)
    cells = []
    for key, _label, dec in TABLE_ROWS:
        row = []
        for n in ctrls:
            mean = _metric_stat(results, n, key)
            std = _metric_stat(results, n, key, "std")
            if math.isnan(mean):
                row.append("—")
            elif math.isnan(std):
                row.append(f"{mean:.{dec}f}")
            else:
                row.append(f"{mean:.{dec}f} ± {std:.{dec}f}")
        cells.append(row)
    table = ax.table(cellText=cells, rowLabels=[r[1] for r in TABLE_ROWS], colLabels=ctrls,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(ctrls))))
    table.scale(1.0, 1.7)
    for j, n in enumerate(ctrls):
        cell = table[(0, j)]
        cell.set_facecolor(_color(n))
        cell.set_text_props(color="white", weight="bold")
    ax.set_title("Özet metrikler (ortalama ± std)", pad=8)
    fig.tight_layout()
    return _save(fig, out)


def make_all_figures(example_trajs: dict[str, dict[str, np.ndarray]], example_scenario,
                     delta_f_all: dict[str, np.ndarray], results: dict,
                     out_dir: str | Path) -> list[Path]:
    """Yedi sekli uretir; Fig1..Fig7 sirasiyla yollari dondurur."""
    _setup_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in FIG_NAMES]
    plot_time_series(example_trajs, paths[0])
    plot_step_zoom(example_trajs, example_scenario, paths[1])
    plot_histogram(delta_f_all, paths[2])
    plot_energy_soc(example_trajs, paths[3])
    plot_shield_strip(example_trajs, paths[4])
    plot_pareto(results, paths[5])
    plot_summary_table(results, paths[6])
    return paths
