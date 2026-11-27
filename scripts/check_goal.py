"""
scripts/check_goal.py — Hedef doğrulayıcı
==========================================
docs/GOAL.md'deki tamamlanma koşulunu makinece denetler.

    python scripts/check_goal.py            # tam denetim (evaluate.py'yi yeniden çalıştırır)
    python scripts/check_goal.py --quick    # pytest'i ve yeniden değerlendirmeyi atlar

Çıkış kodu 0 → hedef tamam. Başarısız her koşul tek satırda yazdırılır.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_CONTROLLERS  # noqa: E402

FIGURES = [
    "Fig1_Zaman_Serisi.png", "Fig2_Yuk_Adimi_Zoom.png", "Fig3_Frekans_Histogram.png",
    "Fig4_Enerji_SOC.png", "Fig5_Kalkan_Seridi.png", "Fig6_Pareto.png", "Fig7_Ozet_Tablo.png",
]
MIN_SCENARIOS = 30
MIN_RL_SEEDS = 3


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="pytest ve yeniden değerlendirmeyi atla")
    args = ap.parse_args()
    passed = []

    # 1) Testler
    if not args.quick:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x"], cwd=ROOT)
        passed.append(check("pytest sıfır hata", r.returncode == 0, f"çıkış kodu {r.returncode}"))

    # 2) results.json
    res_path = ROOT / "results" / "results.json"
    results = None
    if check("results/results.json var", res_path.exists()):
        results = json.loads(res_path.read_text(encoding="utf-8"))
        ctrls = results.get("controllers", {})
        missing = [c for c in ALL_CONTROLLERS if c not in ctrls]
        passed.append(check("7 kontrolcünün tümü mevcut", not missing, f"eksik: {missing}"))
        n = results.get("meta", {}).get("n_scenarios", 0)
        passed.append(check(f"en az {MIN_SCENARIOS} test senaryosu", n >= MIN_SCENARIOS, f"n_scenarios={n}"))
        for rl in ("TD3+Shield", "SAC+Shield"):
            seeds = len(ctrls.get(rl, {}).get("per_seed", {}))
            passed.append(check(f"{rl}: en az {MIN_RL_SEEDS} tohum", seeds >= MIN_RL_SEEDS, f"{seeds} tohum"))
        sweep = results.get("pareto", {}).get("PI_sweep", [])
        passed.append(check("PI Pareto taraması dolu", len(sweep) >= 6, f"{len(sweep)} nokta"))
        gains = ROOT / "results" / "baseline_gains.json"
        passed.append(check("results/baseline_gains.json var (ayarlanmış referanslar)", gains.exists()))
    else:
        passed.append(False)

    # 3) Grafikler
    missing_figs = [f for f in FIGURES if not (ROOT / "figures" / f).exists()]
    passed.append(check("figures/ altında 7 grafik", not missing_figs, f"eksik: {missing_figs}"))

    # 4) README tablosu results.json'dan üretilmiş
    if results is not None:
        try:
            from utils.report import render_results_table
            readme = (ROOT / "README.md").read_text(encoding="utf-8")
            ok = "<!-- RESULTS:START -->" in readme and "<!-- RESULTS:END -->" in readme
            if ok:
                block = readme.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
                ok = block.strip() == render_results_table(results).strip()
            passed.append(check("README sonuç tablosu results.json ile birebir aynı", ok,
                                "python evaluate.py --update-readme çalıştırın"))
        except Exception as exc:  # noqa: BLE001
            passed.append(check("README sonuç tablosu denetlenebildi", False, repr(exc)))

    # 5) Değerlendirme tekrarlanabilir
    if not args.quick and results is not None:
        with tempfile.TemporaryDirectory() as tmp:
            from evaluate import run_evaluation
            meta = results["meta"]
            run_evaluation(n_scenarios=meta["n_scenarios"], seed=meta["seed"],
                           model_dir=ROOT / "models", out_dir=tmp, fig_dir=None,
                           gains_path=ROOT / "results" / "baseline_gains.json")
            same = (Path(tmp) / "results.json").read_bytes() == res_path.read_bytes()
        passed.append(check("evaluate.py iki çalıştırmada aynı results.json'ı üretir", same))

    ok = all(passed)
    print("\nHEDEF TAMAM" if ok else "\nHEDEF HENÜZ TAMAM DEĞİL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
