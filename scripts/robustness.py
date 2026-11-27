"""Dayanıklılık / dağılım dışı (OOD) değerlendirmesi.

Kontrolcüler YENİDEN eğitilmez ve YENİDEN ayarlanmaz; yalnızca tesis (plant) değişir.
Her vaka aynı test bankasını ve aynı kontrolcü kümesini kullanır.

Vakalar:
  nominal  : değişiklik yok
  H_x0.5   : atalet yarıya iner         H_x1.5 : atalet 1,5 katına çıkar
  D_x0.5   : yük sönümü yarıya iner
  T_t_x2   : türbin zaman sabiti iki katına çıkar (daha yavaş türbin)
  load_x1.5: senaryolardaki d_load dizisinin TAMAMI 1,5 ile çarpılır (adım + gürültü);
             meta["load_steps"] büyüklükleri de aynı oranda ölçeklenir. Kırpma yapılmaz:
             ölçekli adımlar üretecin izin verdiği aralığı aşabilir (bilerek: OOD).
  ramp_x0.5: BESS komut hız sınırı yarıya iner

Kullanım: python scripts/robustness.py --n-scenarios 20 --out results/robustness.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import (  # noqa: E402
    DroopController, NoBessController, PIController, PIDController, RLController,
)
from agents.shield import ShieldedEnv  # noqa: E402
from config import (  # noqa: E402
    ALL_CONTROLLERS, CTRL_DROOP, CTRL_NOBESS, CTRL_PI, CTRL_PID, BessParams, EvalParams,
    GridParams, RewardWeights, Scenario, ShieldParams,
)
from envs.microgrid_lfc_env import MicrogridLFCEnv  # noqa: E402
from envs.scenarios import default_generator, make_scenario_bank  # noqa: E402
from evaluate import (  # noqa: E402
    BOOTSTRAP_SEED, RL_CONTROLLERS, RL_SPECS, _clean, _find_models, _load_gains, _nanmean,
)
from utils.metrics import aggregate, compute_metrics  # noqa: E402
from utils.rollout import run_episode  # noqa: E402

# Bu betiğe özgü değerlendirme sabitleri (config.py'ye dokunulmaz)
UNSTABLE_BAND_HZ = 0.05     # son pencerede bu sapma aşılırsa "kararsız" sayılır [Hz]
UNSTABLE_WINDOW_S = 20.0    # kararsızlık için bakılan son pencere [s]
LOAD_SCALE = 1.5

# vaka -> çarpanlar: "grid"/"bess" alan çarpanları, "load" d_load çarpanı
CASES: dict[str, dict] = {
    "nominal": {},
    "H_x0.5": {"grid": {"H": 0.5}},
    "H_x1.5": {"grid": {"H": 1.5}},
    "D_x0.5": {"grid": {"D": 0.5}},
    "T_t_x2": {"grid": {"T_t": 2.0}},
    "load_x1.5": {"load": LOAD_SCALE},
    "ramp_x0.5": {"bess": {"ramp_rate": 0.5}},
}
METRIC_KEYS = ("rmse_mhz", "max_dev_mhz", "bess_throughput_kwh", "autonomy_pct")


def _scaled(params, factors: dict):
    """Dataclass alanlarını çarpanla ölçekleyerek yeni kopya üretir (dataclasses.replace)."""
    return dataclasses.replace(
        params, **{k: getattr(params, k) * f for k, f in factors.items()})


def _scale_load(scenario: Scenario, factor: float) -> Scenario:
    """d_load'un tamamını ölçekler; meta'daki yük adımı büyüklükleri de ölçeklenir."""
    meta = dict(scenario.meta)
    meta["load_steps"] = [(int(k), float(s) * factor) for k, s in scenario.meta.get("load_steps", [])]
    return dataclasses.replace(scenario, d_load=np.asarray(scenario.d_load) * factor, meta=meta)


def _final_window_max_hz(traj: dict, dt: float) -> float:
    """Son UNSTABLE_WINDOW_S saniyedeki en büyük |Δf| [Hz]."""
    n = int(round(UNSTABLE_WINDOW_S / dt))
    return float(np.max(np.abs(traj["delta_f_hz"][-n:])))


def _plot(results: dict, fig_path: Path) -> None:
    """Vaka başına kontrolcü RMSE'sini gruplu çubuk olarak çizer."""
    import matplotlib.pyplot as plt
    cases = list(results)
    ctrls = [c for c in ALL_CONTROLLERS if any(c in results[case] for case in cases)]
    width = 0.8 / max(len(ctrls), 1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, c in enumerate(ctrls):
        vals = [results[case].get(c, {}).get("rmse_mhz", {}).get("mean") for case in cases]
        ys = [np.nan if v is None else v for v in vals]
        ax.bar(np.arange(len(cases)) + i * width, ys, width, label=c)
    ax.set_xticks(np.arange(len(cases)) + width * (len(ctrls) - 1) / 2)
    ax.set_xticklabels(cases)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE [mHz] (log ölçek)")
    ax.set_xlabel("Tesis değişimi (kontrolcüler yeniden ayarlanmadı)")
    ax.set_title("Dayanıklılık: parametre kaymasında frekans RMSE değeri")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def run_robustness(n_scenarios: int = 20, seed: int = EvalParams.seed,
                   model_dir: str | Path = "models",
                   gains_path: str | Path | None = "results/baseline_gains.json",
                   out_path: str | Path = "results/robustness.json",
                   fig_path: str | Path | None = "figures/Fig9_Dayaniklilik.png") -> dict:
    """Tüm vakaları ve kontrolcüleri çalıştırır; sonucu JSON'a yazar ve sözlük olarak döndürür."""
    ev, grid0, bess0 = EvalParams(), GridParams(), BessParams()
    shield, reward = ShieldParams(), RewardWeights()
    gen = default_generator("test")
    bank = make_scenario_bank(gen.wind_levels, n_scenarios, seed)
    gains = _load_gains(Path(gains_path) if gains_path is not None else None)

    # RL modelleri bir kez yüklenir (vakalar arası paylaşılır)
    rl_models: dict[str, list] = {}
    for name in RL_CONTROLLERS:
        algo, pattern, _ = RL_SPECS[name]
        files = _find_models(Path(model_dir), pattern)
        if not files:
            print(f"UYARI: {name} için {model_dir} içinde model bulunamadı; atlanıyor.")
            continue
        from stable_baselines3 import SAC, TD3
        cls = TD3 if algo == "td3" else SAC
        rl_models[name] = [RLController(cls.load(p, device="cpu"), name) for _, p in files]

    results: dict[str, dict] = {}
    for case, spec in CASES.items():
        grid = _scaled(grid0, spec.get("grid", {}))
        bess = _scaled(bess0, spec.get("bess", {}))
        scens = [_scale_load(s, spec["load"]) for s in bank] if "load" in spec else bank

        def raw_env():
            return MicrogridLFCEnv(scenario_generator=gen, grid=grid, bess=bess, reward=reward)

        runs = [
            (CTRL_NOBESS, raw_env(), [NoBessController()]),
            (CTRL_DROOP, raw_env(), [DroopController(gains["droop_k"])]),
            (CTRL_PI, raw_env(), [PIController(gains["pi_kp"], gains["pi_ki"], grid.dt)]),
            (CTRL_PID, raw_env(), [PIDController(gains["pid_kp"], gains["pid_ki"],
                                                 gains["pid_kd"], grid.dt)]),
        ]
        for name, ctrls in rl_models.items():
            env = ShieldedEnv(raw_env(), shield) if RL_SPECS[name][2] else raw_env()
            runs.append((name, env, ctrls))

        results[case] = {}
        for name, env, ctrls in runs:
            per_seed = []   # tohum -> metrik -> senaryo listesi
            for ctrl in ctrls:
                pm: dict[str, list[float]] = {k: [] for k in (*METRIC_KEYS, "final_max_hz")}
                for scen in scens:
                    traj = run_episode(env, ctrl, scen)
                    m = compute_metrics(traj, scen, grid, bess, ev.settle_band_hz)
                    for k in METRIC_KEYS:
                        pm[k].append(m[k])
                    pm["final_max_hz"].append(_final_window_max_hz(traj, grid.dt))
                per_seed.append(pm)
            # tohumlar üzerinden senaryo başına ortalama (evaluate.py gibi)
            per_scen = {k: [_nanmean([s[k][i] for s in per_seed]) for i in range(len(scens))]
                        for k in per_seed[0]}
            entry = {k: aggregate(per_scen[k], ev.bootstrap_samples, seed=BOOTSTRAP_SEED)
                     for k in METRIC_KEYS}
            # kararsız: tohum-ortalamalı son-pencere tepe sapması eşiği aşan senaryo sayısı
            entry["unstable"] = int(sum(v > UNSTABLE_BAND_HZ for v in per_scen["final_max_hz"]))
            results[case][name] = entry

    results = _clean(results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, sort_keys=True, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    if fig_path is not None:
        _plot(results, Path(fig_path))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Dayanıklılık (OOD) değerlendirmesi.")
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--seed", type=int, default=EvalParams.seed)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--gains", default="results/baseline_gains.json")
    ap.add_argument("--out", default="results/robustness.json")
    ap.add_argument("--fig", default="figures/Fig9_Dayaniklilik.png")
    a = ap.parse_args()
    res = run_robustness(a.n_scenarios, a.seed, a.model_dir, a.gains, a.out, a.fig)
    for case, ctrls in res.items():
        print(case, {c: (None if v["rmse_mhz"]["mean"] is None else round(v["rmse_mhz"]["mean"], 1))
                     for c, v in ctrls.items()})


if __name__ == "__main__":
    main()
