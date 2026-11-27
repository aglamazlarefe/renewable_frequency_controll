"""Sonuç tablosu üretimi ve README güncelleme (SPEC 3.13)."""

from __future__ import annotations

from pathlib import Path

from config import ALL_CONTROLLERS

MARKER_START = "<!-- RESULTS:START -->"
MARKER_END = "<!-- RESULTS:END -->"
MISSING = "—"
PM = "±"

# (metrik anahtarı, etiket, ondalık basamak)
TABLE_ROWS = (
    ("rmse_mhz", "RMSE [mHz]", 1),
    ("max_dev_mhz", "Maks. sapma [mHz]", 1),
    ("band_100_pct", r"Süre \|Δf\|<100 mHz [%]", 1),  # | Markdown tablosunda kaçışlı
    ("settling_time_s", "Yerleşme süresi [s]", 1),
    ("bess_throughput_kwh", "BESS enerji geçişi [kWh]", 2),
    ("control_jitter", "Kontrol titreşimi", 1),
    ("autonomy_pct", "Ajan özerkliği [%]", 1),
    ("mean_cost", "Ortalama maliyet", 3),
)


def _cell(metrics: dict, key: str, decimals: int) -> str:
    """'ortalama ± std' hücresi; değer eksikse em dash."""
    entry = metrics.get(key)
    if not entry:
        return MISSING
    mean, std = entry.get("mean"), entry.get("std")
    if mean is None:
        return MISSING
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} {PM} {std:.{decimals}f}"


def render_results_table(results: dict) -> str:
    """results sözlüğünden deterministik Markdown tablo üretir."""
    present = results["controllers"]
    names = [c for c in ALL_CONTROLLERS if c in present]
    lines = ["| Metrik | " + " | ".join(names) + " |",
             "|---|" + "|".join("---" for _ in names) + "|"]
    for key, label, decimals in TABLE_ROWS:
        cells = [_cell(present[n]["metrics"], key, decimals) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def update_readme(results: dict, readme_path: str | Path = "README.md") -> None:
    """README'deki işaretçiler arasını tabloyla değiştirir; işaretçi yoksa ValueError."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    start = text.find(MARKER_START)
    end = text.find(MARKER_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"{path} içinde {MARKER_START} / {MARKER_END} işaretçileri bulunamadı")
    table = render_results_table(results)
    new = text[:start + len(MARKER_START)] + "\n" + table + text[end:]
    path.write_text(new, encoding="utf-8")
