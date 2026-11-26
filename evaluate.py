"""Değerlendirme (SPEC 3.12): tüm kontrolcüler test bankasında birebir aynı senaryolarla çalışır."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np

from agents.baselines import (
    DroopController, NoBessController, PIController, PIDController, RLController,
)
from agents.shield import ShieldedEnv
from config import (
    ALL_CONTROLLERS, CTRL_DROOP, CTRL_NOBESS, CTRL_PI, CTRL_PID, CTRL_SAC_SHIELD,
    CTRL_TD3_RAW, CTRL_TD3_SHIELD, BaselineGains, BessParams, EvalParams, GridParams,
    RewardWeights, ShieldParams,
)
from envs.microgrid_lfc_env import MicrogridLFCEnv
from envs.scenarios import default_generator, make_scenario_bank
from utils.metrics import aggregate, compute_metrics, paired_test
from utils.rollout import run_episode

RL_CONTROLLERS = (CTRL_TD3_SHIELD, CTRL_SAC_SHIELD, CTRL_TD3_RAW)
# kontrolcü adı -> (algoritma, dosya deseni, kalkanlı mı)
RL_SPECS = {
    CTRL_TD3_SHIELD: ("td3", re.compile(r"^td3_seed(\d+)\.zip$"), True),
    CTRL_SAC_SHIELD: ("sac", re.compile(r"^sac_seed(\d+)\.zip$"), True),
    CTRL_TD3_RAW: ("td3", re.compile(r"^td3_noshield_seed(\d+)\.zip$"), False),
}
CSV_STATS = ("mean", "std", "ci_low", "ci_high")
BOOTSTRAP_SEED = 0   # bootstrap tohumu sabit: sonuç tekrarlanabilir


def _clean(x):
    """JSON-temiz hale getirir: NaN -> None, numpy -> Python, tuple -> list, anahtarlar str."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def _find_models(model_dir: Path, pattern: re.Pattern) -> list[tuple[int, Path]]:
    """Desene tam uyan dosyaları (tohum, yol) olarak tohuma göre sıralı döndürür."""
    found = []
    if model_dir.is_dir():
        for p in model_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    return sorted(found)


def _load_gains(gains_path: Path | None) -> dict:
    """Varsayılan kazançlar; gains_path varsa dosyadaki değerlerle üzerine yazılır."""
    gains = dataclasses.asdict(BaselineGains())
    if gains_path is None:
        return gains
    if not gains_path.exists():
        print(f"UYARI: {gains_path} bulunamadı; klasik kontrolcüler AYARLANMAMIŞ varsayılan kazançlarla "
              f"çalışıyor. Önce `python tune_baselines.py` çalıştırın.")
        return gains
    loaded = json.loads(gains_path.read_text(encoding="utf-8"))
    unknown, missing = set(loaded) - set(gains), set(gains) - set(loaded)
    if unknown or missing:
        raise ValueError(f"{gains_path}: bilinmeyen anahtarlar {sorted(unknown)}, eksik anahtarlar {sorted(missing)}")
    gains.update({k: float(v) for k, v in loaded.items()})
    return gains


def _nanmean(values) -> float:
    """NaN/None dışlanarak ortalama; geçerli değer yoksa NaN."""
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    ok = arr[~np.isnan(arr)]
    return float(ok.mean()) if ok.size else math.nan


def _first_step_size(scenario) -> float:
    """Senaryodaki ilk yük adımının mutlak büyüklüğü [pu]; adım yoksa 0."""
    steps = scenario.meta.get("load_steps") or []
    if not steps:
        return 0.0
    return abs(float(sorted(steps, key=lambda s: s[0])[0][1]))


def run_evaluation(n_scenarios: int = EvalParams.n_scenarios, seed: int = EvalParams.seed,
                   model_dir: str | Path = "models", out_dir: str | Path = "results",
                   fig_dir: str | Path | None = "figures",
                   gains_path: str | Path | None = "results/baseline_gains.json") -> dict:
    """Tüm kontrolcüleri aynı test senaryolarında çalıştırıp results sözlüğünü üretir ve yazar."""
    from stable_baselines3 import SAC, TD3

    ev, grid, bess = EvalParams(), GridParams(), BessParams()
    shield, reward = ShieldParams(), RewardWeights()
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    gains_file = Path(gains_path) if gains_path is not None else None
    gen = default_generator("test")
    bank = make_scenario_bank(gen.wind_levels, n_scenarios, seed)
    gains = _load_gains(gains_file)

    def raw_env():
        return MicrogridLFCEnv(scenario_generator=gen, grid=grid, bess=bess, reward=reward)

    def shielded_env():
        return ShieldedEnv(raw_env(), shield)

    # (ad, env, [(tohum veya None, kontrolcü)])
    runs: list[tuple[str, object, list]] = [
        (CTRL_NOBESS, raw_env(), [(None, NoBessController())]),
        (CTRL_DROOP, raw_env(), [(None, DroopController(gains["droop_k"]))]),
        (CTRL_PI, raw_env(), [(None, PIController(gains["pi_kp"], gains["pi_ki"], grid.dt))]),
        (CTRL_PID, raw_env(), [(None, PIDController(gains["pid_kp"], gains["pid_ki"],
                                                    gains["pid_kd"], grid.dt))]),
    ]
    for name in RL_CONTROLLERS:
        algo, pattern, shielded = RL_SPECS[name]
        files = _find_models(model_dir, pattern)
        if not files:
            print(f"UYARI: {name} için {model_dir} içinde model bulunamadı; atlanıyor.")
            continue
        cls = TD3 if algo == "td3" else SAC
        ctrls = [(s, RLController(cls.load(p, device="cpu"), name)) for s, p in files]
        runs.append((name, shielded_env() if shielded else raw_env(), ctrls))

    example_idx = int(np.argmax([_first_step_size(s) for s in bank]))
    controllers: dict[str, dict] = {}
    example_trajs: dict[str, dict] = {}
    delta_f_all: dict[str, np.ndarray] = {}

    for name, env, ctrls in runs:
        seed_metrics: list[dict[str, list[float]]] = []   # tohum -> metrik -> senaryo listesi
        dfs = []
        for ci, (_, ctrl) in enumerate(ctrls):
            per_metric: dict[str, list[float]] = {}
            for si, scen in enumerate(bank):
                traj = run_episode(env, ctrl, scen)
                m = compute_metrics(traj, scen, grid, bess, ev.settle_band_hz)
                for k, v in m.items():
                    per_metric.setdefault(k, []).append(v)
                dfs.append(traj["delta_f_hz"])
                if ci == 0 and si == example_idx:
                    example_trajs[name] = traj
            seed_metrics.append(per_metric)
        keys = list(seed_metrics[0])
        per_scenario = {k: [_nanmean([sm[k][i] for sm in seed_metrics])
                            for i in range(len(bank))] for k in keys}
        entry = {
            "metrics": {k: aggregate(per_scenario[k], ev.bootstrap_samples, seed=BOOTSTRAP_SEED)
                        for k in keys},
            "per_scenario": per_scenario,
        }
        if name in RL_CONTROLLERS:
            entry["per_seed"] = {str(s): {k: _nanmean(sm[k]) for k in keys}
                                 for (s, _), sm in zip(ctrls, seed_metrics)}
        controllers[name] = entry
        delta_f_all[name] = np.concatenate(dfs)

    comparisons: dict[str, dict] = {}
    for rl in RL_CONTROLLERS:
        if rl not in controllers:
            continue
        for base in (CTRL_PI, CTRL_PID):
            comparisons[f"{rl}_vs_{base}"] = {
                k: paired_test(controllers[rl]["per_scenario"][k],
                               controllers[base]["per_scenario"][k])
                for k in controllers[rl]["per_scenario"]}

    sweep: list = []
    if gains_file is not None:
        pareto_file = gains_file.parent / "pareto_pi.json"
        if pareto_file.exists():
            sweep = json.loads(pareto_file.read_text(encoding="utf-8"))

    results = _clean({
        "meta": {"n_scenarios": n_scenarios, "seed": seed,
                 "grid": dataclasses.asdict(grid), "bess": dataclasses.asdict(bess),
                 "shield": dataclasses.asdict(shield), "reward": dataclasses.asdict(reward),
                 "gains": gains},
        "controllers": controllers,
        "comparisons": comparisons,
        "pareto": {"PI_sweep": sweep},
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = ["controller,metric,mean,std,ci_low,ci_high"]
    for name in ALL_CONTROLLERS:
        if name not in results["controllers"]:
            continue
        for k, st in sorted(results["controllers"][name]["metrics"].items()):
            cells = ["" if st[s] is None else repr(st[s]) for s in CSV_STATS]
            rows.append(",".join([name, k, *cells]))
    (out_dir / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if fig_dir is not None:
        from utils.plotting import make_all_figures
        make_all_figures(example_trajs, bank[example_idx], delta_f_all, results, fig_dir)
    return results


def main() -> None:
    ev = EvalParams()
    ap = argparse.ArgumentParser(description="Kontrolcüleri test bankasında değerlendirir.")
    ap.add_argument("--n-scenarios", type=int, default=ev.n_scenarios)
    ap.add_argument("--seed", type=int, default=ev.seed)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()
    from utils.report import render_results_table, update_readme
    results = run_evaluation(args.n_scenarios, args.seed, args.model_dir)
    print(render_results_table(results))
    if args.update_readme:
        update_readme(results, "README.md")


if __name__ == "__main__":
    main()
