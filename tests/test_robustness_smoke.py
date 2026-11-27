"""Dayanıklılık betiği duman testi: yapı, belirlilik ve nominal PI tutarlılığı."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robustness import CASES, run_robustness  # noqa: E402


def _run(tmp_path: Path, name: str) -> tuple[dict, bytes]:
    out = tmp_path / name
    res = run_robustness(n_scenarios=1, model_dir=tmp_path / "bos_modeller", gains_path=None,
                         out_path=out, fig_path=None)
    return res, out.read_bytes()


def test_robustness_structure_determinism_and_nominal(tmp_path):
    res1, b1 = _run(tmp_path, "a.json")
    _, b2 = _run(tmp_path, "b.json")
    assert b1 == b2, "iki çalıştırma bayt düzeyinde aynı olmalı"
    assert json.loads(b1) == res1

    assert list(res1) == list(CASES)
    for ctrls in res1.values():
        assert {"NoBESS", "Droop", "PI", "PID"} <= set(ctrls)
        for entry in ctrls.values():
            assert set(entry) == {"rmse_mhz", "max_dev_mhz", "bess_throughput_kwh",
                                  "autonomy_pct", "unstable"}
            assert set(entry["rmse_mhz"]) == {"mean", "std", "ci_low", "ci_high"}
            assert isinstance(entry["unstable"], int)

    # nominal PI = aynı senaryoda nominal ortamda utils.metrics
    from agents.baselines import PIController
    from config import BaselineGains, BessParams, EvalParams, GridParams
    from envs.microgrid_lfc_env import MicrogridLFCEnv
    from envs.scenarios import default_generator, make_scenario_bank
    from utils.metrics import compute_metrics
    from utils.rollout import run_episode

    gen = default_generator("test")
    scen = make_scenario_bank(gen.wind_levels, 1, EvalParams.seed)[0]
    g, b, k = GridParams(), BessParams(), BaselineGains()
    env = MicrogridLFCEnv(scenario_generator=gen, grid=g, bess=b)
    m = compute_metrics(run_episode(env, PIController(k.pi_kp, k.pi_ki, g.dt), scen),
                        scen, g, b, EvalParams.settle_band_hz)
    pi = res1["nominal"]["PI"]
    assert pi["rmse_mhz"]["mean"] == pytest.approx(m["rmse_mhz"], rel=1e-12)
    assert pi["max_dev_mhz"]["mean"] == pytest.approx(m["max_dev_mhz"], rel=1e-12)
    assert pi["bess_throughput_kwh"]["mean"] == pytest.approx(m["bess_throughput_kwh"], rel=1e-12)
