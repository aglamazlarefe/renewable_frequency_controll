"""Model dogrulama: yuk adimi tablosu, kararlilik taramasi ve rastgele senaryo istatistigi.

Calistirma: python scripts/validate_model.py [--out docs/model_validation_output.txt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.baselines import (DroopController, NoBessController, PIController,  # noqa: E402
                              PIDController)
from agents.shield import ShieldedEnv  # noqa: E402
from config import (BaselineGains, BessParams, EvalParams, GridParams,  # noqa: E402
                    ScenarioParams, ShieldParams, make_step_scenario)
from envs.microgrid_lfc_env import MicrogridLFCEnv  # noqa: E402
from envs.scenarios import default_generator  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402
from utils.rollout import run_episode  # noqa: E402

# Betige ozgu dogrulama ayarlari (fiziksel sabit degil)
STEP_SIZES_PU = (0.02, 0.05, 0.10, 0.15)
STEP_TIME_S = 10.0
SWEEP_KP = tuple(float(k) for k in range(1, 13))
SWEEP_STEP_PU = 0.10
TAIL_S = 20.0            # kararlilik icin son pencere [s]
TAIL_MAX_HZ = 0.05       # son pencerede |df| bunu asarsa "kararsiz/salinimli" say [Hz]
N_RANDOM = 20
RANDOM_SEED = 7


class ZeroController:
    """Sifir aksiyon veren ajan (kalkan testi icin)."""
    name = "ZeroAgent+Shield"

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        return np.zeros(1, dtype=np.float32)


def step_scenario(size: float, grid: GridParams, sp: ScenarioParams):
    return make_step_scenario(sp.n_steps(grid.dt), int(round(STEP_TIME_S / grid.dt)), size)


def row(name, traj, scen, grid, bess, bessp_unused=None):
    m = compute_metrics(traj, scen, grid, bess, EvalParams().settle_band_hz)
    return (name, m["nadir_mhz"], m["settling_time_s"], m["bess_throughput_kwh"],
            float(np.max(np.abs(traj["p_ess"]))), abs(float(traj["delta_f_hz"][-1])) * 1000.0,
            100.0 * float(np.mean(traj["shield_active"])))


def main(out: str | None) -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    grid, bess, sp, g = GridParams(), BessParams(), ScenarioParams(), BaselineGains()
    raw = MicrogridLFCEnv()
    shielded = ShieldedEnv(MicrogridLFCEnv(), ShieldParams())
    ctrls = [("NoBESS", raw, NoBessController()),
             ("Droop", raw, DroopController(g.droop_k)),
             ("PI", raw, PIController(g.pi_kp, g.pi_ki, grid.dt)),
             ("PID", raw, PIDController(g.pid_kp, g.pid_ki, g.pid_kd, grid.dt)),
             ("ZeroAgent+Shield", shielded, ZeroController())]

    emit("## 1. Yuk adimi tablosu (t=10 s, 150 s, gurultusuz)")
    for size in STEP_SIZES_PU:
        scen = step_scenario(size, grid, sp)
        emit(f"\n### Adim = {size:.2f} pu (yuk artisi)\n")
        emit("| Kontrolcu | nadir [mHz] | yerlesme [s] | throughput [kWh] | max abs p_ess [pu] "
             "| son abs df [mHz] | kalkan aktif [%] |")
        emit("|---|---|---|---|---|---|---|")
        for name, env, c in ctrls:
            r = row(name, run_episode(env, c, scen, seed=0), scen, grid, bess)
            emit(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.1f} | {r[6]:.1f} |")

    emit(f"\n## 2. Kararlilik taramasi ({SWEEP_STEP_PU:.2f} pu adim)")
    emit(f"Olcut: son {TAIL_S:.0f} s icinde max abs df < {TAIL_MAX_HZ*1000:.0f} mHz ve sonlu.\n")
    scen = step_scenario(SWEEP_STEP_PU, grid, sp)
    tail_n = int(round(TAIL_S / grid.dt))
    emit("| kp | Droop son-pencere max [mHz] | Droop kararli | PI son-pencere max [mHz] | PI kararli |")
    emit("|---|---|---|---|---|")
    ok = {"Droop": [], "PI": []}
    for kp in SWEEP_KP:
        cells = []
        for key, c in (("Droop", DroopController(kp)), ("PI", PIController(kp, g.pi_ki, grid.dt))):
            tr = run_episode(raw, c, scen, seed=0)
            tail = float(np.max(np.abs(tr["delta_f_hz"][-tail_n:])))
            stable = bool(np.isfinite(tail) and tail < TAIL_MAX_HZ)
            if stable:
                ok[key].append(kp)
            cells.append((tail * 1000.0, stable))
        emit(f"| {kp:.0f} | {cells[0][0]:.1f} | {cells[0][1]} | {cells[1][0]:.1f} | {cells[1][1]} |")
    for key in ok:
        rng_txt = f"{min(ok[key]):.0f} .. {max(ok[key]):.0f}" if ok[key] else "yok"
        emit(f"\nKararli {key} kp araligi: {rng_txt}")

    emit(f"\n## 3. {N_RANDOM} rastgele egitim senaryosu (PI varsayilan, sifir-ajan+kalkan)")
    gen = default_generator("train")
    rng = np.random.default_rng(RANDOM_SEED)
    mx, trig, act_pct = [], [], []
    for _ in range(N_RANDOM):
        scen = gen.sample(rng)
        tr = run_episode(raw, PIController(g.pi_kp, g.pi_ki, grid.dt), scen, seed=0)
        mx.append(float(np.max(np.abs(tr["delta_f_hz"]))) * 1000.0)
        tz = run_episode(shielded, ZeroController(), scen, seed=0)
        trig.append(bool(np.any(tz["shield_active"])))
        act_pct.append(100.0 * float(np.mean(tz["shield_active"])))
    mx = np.array(mx)
    q = np.percentile(mx, [0, 25, 50, 75, 90, 100])
    emit("\nPI max abs df [mHz]: min %.1f, q25 %.1f, medyan %.1f, q75 %.1f, q90 %.1f, max %.1f" % tuple(q))
    emit(f"PI ile max abs df > engage ({ShieldParams().engage_hz*1000:.0f} mHz) olan senaryo: "
         f"{int(np.sum(mx > ShieldParams().engage_hz*1000))}/{N_RANDOM}")
    emit(f"Sifir-ajan kalkani tetikledi: {int(sum(trig))}/{N_RANDOM}; ortalama aktif sure %{np.mean(act_pct):.1f}")
    if out:
        Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    main(ap.parse_args().out)
