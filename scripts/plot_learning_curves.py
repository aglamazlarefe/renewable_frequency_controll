"""Öğrenme eğrileri şekli (Fig8) ve özet JSON.

train.py'nin yazdığı TensorBoard olay dosyalarını (logs/<model>_<n>/) okur:
  eval/mean_reward, episode/ep_rmse_hz, episode/ep_autonomy.
Algoritma başına tohumlar üzerinden ortalama çizgisi ve min-maks gölgesi çizer.
Eğitim sürerken yarım yazılmış günlüklerle de çalışır; günlüğü olmayan model atlanır.

Kullanım (proje kökünden):
    PY scripts/plot_learning_curves.py [--logs logs] [--out figures/Fig8_Ogrenme_Egrileri.png]
                                       [--json results/learning_curves.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CTRL_SAC_SHIELD, CTRL_TD3_RAW, CTRL_TD3_SHIELD  # noqa: E402
from utils.plotting import CONTROLLER_COLORS, CONTROLLER_STYLES, DPI  # noqa: E402

TAG_REWARD = "eval/mean_reward"
TAG_RMSE = "episode/ep_rmse_hz"
TAG_AUTONOMY = "episode/ep_autonomy"
_MHZ_PER_HZ = 1000.0
_PCT = 100.0
SEEDS = (0, 1, 2)
SMOOTH_WINDOW = 3  # bölüm metrikleri için hareketli ortalama penceresi [nokta]

# Denetleyici adı -> (günlük adı kalıbı, Türkçe etiket)
GROUPS = {
    CTRL_TD3_SHIELD: ("td3_seed{seed}", "TD3 + Kalkan"),
    CTRL_SAC_SHIELD: ("sac_seed{seed}", "SAC + Kalkan"),
    CTRL_TD3_RAW: ("td3_noshield_seed{seed}", "TD3 kalkansız"),
}


def latest_run_dir(logs: Path, model: str) -> Path | None:
    """logs/<model>_<n> içinde en büyük n'li dizin (en yeni koşu); yoksa None."""
    pat = re.compile(rf"^{re.escape(model)}_(\d+)$")
    found = [(int(m.group(1)), p) for p in logs.iterdir()
             if p.is_dir() and (m := pat.match(p.name))] if logs.is_dir() else []
    return max(found, key=lambda t: t[0])[1] if found else None


def load_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Dizindeki tüm skaler etiketler -> (adımlar, değerler)."""
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        ev = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev], dtype=float),
                    np.array([e.value for e in ev], dtype=float))
    return out


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Kenar etkisi olmayan (kısalan pencereli) hareketli ortalama."""
    if window <= 1 or y.size < 2:
        return y
    return np.array([y[max(0, i - window + 1): i + 1].mean() for i in range(y.size)])


def aggregate(series: list[tuple[np.ndarray, np.ndarray]]):
    """Tohumları adım ekseninde hizalar: (adımlar, ortalama, min, maks); dışsal kestirim yok."""
    series = [(x, y) for x, y in series if x.size]
    if not series:
        return None
    grid = np.unique(np.concatenate([x for x, _ in series]))
    stack = np.full((len(series), grid.size), np.nan)
    for i, (x, y) in enumerate(series):
        inside = (grid >= x.min()) & (grid <= x.max())
        stack[i, inside] = np.interp(grid[inside], x, y)
    return grid, np.nanmean(stack, 0), np.nanmin(stack, 0), np.nanmax(stack, 0)


def summarize_reward(x: np.ndarray, y: np.ndarray) -> dict:
    best = int(np.argmax(y))
    return {"final_reward": float(y[-1]), "best_reward": float(y[best]),
            "best_step": int(x[best]), "last_step": int(x[-1])}


def collect(logs: Path):
    """Grup -> {tag: [(x, y) ...]} ve model başına JSON özeti."""
    data: dict[str, dict[str, list]] = {}
    summary: dict[str, dict] = {}
    for ctrl, (pattern, _label) in GROUPS.items():
        for seed in SEEDS:
            model = pattern.format(seed=seed)
            run = latest_run_dir(logs, model)
            if run is None:
                print(f"UYARI: {model} için günlük bulunamadı, atlanıyor.", file=sys.stderr)
                continue
            scalars = load_scalars(run)
            if not scalars:
                print(f"UYARI: {run} içinde skaler yok, atlanıyor.", file=sys.stderr)
                continue
            if TAG_REWARD in scalars:
                summary[model] = summarize_reward(*scalars[TAG_REWARD])
            else:
                print(f"UYARI: {model} henüz {TAG_REWARD} yazmamış.", file=sys.stderr)
            for tag in (TAG_REWARD, TAG_RMSE, TAG_AUTONOMY):
                if tag in scalars:
                    x, y = scalars[tag]
                    if tag != TAG_REWARD:
                        y = smooth(y, SMOOTH_WINDOW)
                    data.setdefault(ctrl, {}).setdefault(tag, []).append((x, y))
    return data, summary


def make_figure(data: dict, out: Path) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    })
    panels = (
        (TAG_REWARD, "(a) Doğrulama ödülü", "Ortalama bölüm ödülü [-]", 1.0),
        (TAG_RMSE, "(b) Bölüm RMSE", "RMSE [mHz]", _MHZ_PER_HZ),
        (TAG_AUTONOMY, "(c) Kalkan özerkliği", "Özerklik [%]", _PCT),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    handles = {}
    for ax, (tag, title, ylabel, scale) in zip(axes, panels):
        for ctrl, (_p, label) in GROUPS.items():
            agg = aggregate(data.get(ctrl, {}).get(tag, []))
            if agg is None:
                continue
            x, mean, lo, hi = agg
            color = CONTROLLER_COLORS[ctrl]
            ax.fill_between(x, lo * scale, hi * scale, color=color, alpha=0.18, lw=0)
            (line,) = ax.plot(x, mean * scale, color=color, ls=CONTROLLER_STYLES[ctrl],
                              lw=1.8, label=label)
            handles[label] = line
        ax.set_title(title)
        ax.set_xlabel("Ortam adımı")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="x", style="plain")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    axes[2].set_ylim(top=105)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles),
                   frameon=False)
    fig.suptitle("Öğrenme eğrileri (çizgi: tohum ortalaması, gölge: min-maks)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, default=ROOT / "logs")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "Fig8_Ogrenme_Egrileri.png")
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "learning_curves.json")
    args = ap.parse_args()

    data, summary = collect(args.logs)
    if not data:
        print("HATA: hiçbir modelde günlük yok; şekil üretilmedi.", file=sys.stderr)
        return 1
    make_figure(data, args.out)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(dict(sorted(summary.items())), indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"Yazıldı: {args.out}\nYazıldı: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
