"""Klasik referans kontrolcülerin (Droop, PI, PID) Optuna ile ayarı ve PI Pareto taraması (SPEC §3.11).

- tune: EĞİTİM bölmesinden sabit senaryo bankası; amaç = bölüm ortalaması -reward (RL ile aynı maliyet).
- pareto_sweep: TEST bölmesinde PI kazanç ızgarası; rmse_mhz ve bess_throughput_kwh.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna

from agents.baselines import DroopController, PIController, PIDController
from config import BaselineGains, EvalParams, GridParams, Scenario
from envs.microgrid_lfc_env import MicrogridLFCEnv
from envs.scenarios import default_generator, make_scenario_bank
from utils.metrics import compute_metrics
from utils.rollout import run_episode


@dataclass(frozen=True)
class SearchSpace:
    """Arama aralıkları (docs/MODEL_VALIDATION.md, kararlı bölge)."""
    droop_k: tuple[float, float] = (0.5, 9.0)
    pi_kp: tuple[float, float] = (0.5, 8.0)
    pi_ki: tuple[float, float] = (0.1, 5.0)
    pid_kp: tuple[float, float] = (0.5, 8.0)
    pid_ki: tuple[float, float] = (0.1, 5.0)
    pid_kd: tuple[float, float] = (0.0, 2.0)


@dataclass(frozen=True)
class ParetoGrid:
    """PI Pareto ızgarası; kp değerleri kararlı aralık (1..8) içindedir."""
    kp: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    ki: tuple[float, ...] = (0.5, 2.0, 4.0)


SPACE = SearchSpace()
GRID = ParetoGrid()


def _write_json(obj, out_path: str | Path) -> None:
    """JSON yazar; üst dizini gerekirse oluşturur."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _mean_cost(controller, bank: list[Scenario], env: MicrogridLFCEnv) -> float:
    """Bankadaki bölümlerin ortalama maliyeti (-mean reward) üzerinden ortalaması."""
    costs = [-float(np.mean(run_episode(env, controller, sc)["reward"])) for sc in bank]
    return float(np.mean(costs))


def _make_env(split: str) -> MicrogridLFCEnv:
    """Kalkansız ham ortam; senaryolar options ile verilir."""
    return MicrogridLFCEnv(scenario_generator=default_generator(split))


def _train_bank(n_scenarios: int, seed: int) -> list[Scenario]:
    gen = default_generator("train")
    return make_scenario_bank(gen.wind_levels, n_scenarios, seed)


def gains_cost(gains: dict[str, float], n_scenarios: int = 20, seed: int = 0,
               split: str = "train") -> dict[str, float]:
    """Verilen altı kazancın kontrolcü başına bank maliyeti (CLI karşılaştırması için)."""
    grid = GridParams()
    gen = default_generator(split)
    bank = make_scenario_bank(gen.wind_levels, n_scenarios, seed)
    env = _make_env(split)
    return {
        "Droop": _mean_cost(DroopController(gains["droop_k"]), bank, env),
        "PI": _mean_cost(PIController(gains["pi_kp"], gains["pi_ki"], grid.dt), bank, env),
        "PID": _mean_cost(PIDController(gains["pid_kp"], gains["pid_ki"], gains["pid_kd"], grid.dt),
                          bank, env),
    }


def _optimize(objective, n_trials: int, seed: int) -> optuna.Study:
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def tune(n_trials: int = 150, n_scenarios: int = 20, seed: int = 0,
         out_path: str | Path = "results/baseline_gains.json") -> dict:
    """Droop, PI ve PID kazançlarını eğitim bankasında Optuna ile ayarlar."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dt = GridParams().dt
    bank = _train_bank(n_scenarios, seed)
    env = _make_env("train")

    def droop_obj(trial: optuna.Trial) -> float:
        k = trial.suggest_float("droop_k", *SPACE.droop_k)
        return _mean_cost(DroopController(k), bank, env)

    def pi_obj(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("pi_kp", *SPACE.pi_kp)
        ki = trial.suggest_float("pi_ki", *SPACE.pi_ki)
        return _mean_cost(PIController(kp, ki, dt), bank, env)

    def pid_obj(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("pid_kp", *SPACE.pid_kp)
        ki = trial.suggest_float("pid_ki", *SPACE.pid_ki)
        kd = trial.suggest_float("pid_kd", *SPACE.pid_kd)
        return _mean_cost(PIDController(kp, ki, kd, dt), bank, env)

    droop = _optimize(droop_obj, n_trials, seed).best_params
    pi = _optimize(pi_obj, n_trials, seed).best_params
    pid = _optimize(pid_obj, n_trials, seed).best_params

    gains = {
        "droop_k": float(droop["droop_k"]),
        "pi_kp": float(pi["pi_kp"]),
        "pi_ki": float(pi["pi_ki"]),
        "pid_kp": float(pid["pid_kp"]),
        "pid_ki": float(pid["pid_ki"]),
        "pid_kd": float(pid["pid_kd"]),
    }
    _write_json(gains, out_path)
    return gains


def pareto_sweep(n_scenarios: int = EvalParams.n_scenarios, seed: int = EvalParams.seed,
                 out_path: str | Path = "results/pareto_pi.json") -> list[dict]:
    """PI kp x ki ızgarasını TEST bankasında değerlendirir.

    Varsayılan banka evaluate.py ile aynıdır (EvalParams), böylece Şekil 6'daki eğri ve kontrolcü
    noktaları aynı senaryolar üzerinden hesaplanır.
    """
    dt = GridParams().dt
    gen = default_generator("test")
    bank = make_scenario_bank(gen.wind_levels, n_scenarios, seed)
    env = _make_env("test")
    rows: list[dict] = []
    for kp in GRID.kp:
        for ki in GRID.ki:
            ctrl = PIController(kp, ki, dt)
            metrics = [compute_metrics(run_episode(env, ctrl, sc), sc) for sc in bank]
            rows.append({
                "kp": float(kp),
                "ki": float(ki),
                "rmse_mhz": float(np.mean([m["rmse_mhz"] for m in metrics])),
                "bess_throughput_kwh": float(np.mean([m["bess_throughput_kwh"] for m in metrics])),
            })
    _write_json(rows, out_path)
    return rows


def main() -> None:
    """Ayarı ve Pareto taramasını varsayılan boyutlarla çalıştırır, iyileşmeyi yazdırır."""
    gains = tune()
    pareto_sweep()
    default = asdict(BaselineGains())
    cost_default = gains_cost(default)
    cost_tuned = gains_cost(gains)
    print("Kazanç                varsayılan    ayarlı")
    for name in default:
        print(f"{name:<18} {default[name]:>10.3f} {gains[name]:>10.3f}")
    print("\nEğitim bankası ortalama maliyeti (düşük iyi)")
    for ctrl in cost_default:
        print(f"{ctrl:<8} varsayılan {cost_default[ctrl]:.4f}  ayarlı {cost_tuned[ctrl]:.4f}")


if __name__ == "__main__":
    main()
