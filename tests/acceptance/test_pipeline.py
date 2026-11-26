"""Kabul testleri — eğitim, kazanç ayarı, değerlendirme ve rapor (SPEC §3.10–3.13).

Kısa "duman" koşuları: amaç öğrenme kalitesi değil, boru hattının uçtan uca çalışması.
BU DOSYA SÖZLEŞMEDİR: değiştirmeyin. Yanlış olduğunu düşünüyorsanız gerekçesiyle raporlayın.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from config import CTRL_DROOP, CTRL_NOBESS, CTRL_PI, CTRL_PID, CTRL_SAC_SHIELD, CTRL_TD3_RAW, CTRL_TD3_SHIELD

pytestmark = pytest.mark.slow

FIGURES = [
    "Fig1_Zaman_Serisi.png", "Fig2_Yuk_Adimi_Zoom.png", "Fig3_Frekans_Histogram.png",
    "Fig4_Enerji_SOC.png", "Fig5_Kalkan_Seridi.png", "Fig6_Pareto.png", "Fig7_Ozet_Tablo.png",
]
SMOKE = dict(learning_starts=100, batch_size=64, buffer_size=5000)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """TD3 (kalkanlı, 2 tohum), SAC (kalkanlı) ve TD3 (kalkansız) için çok kısa eğitimler."""
    from train import train
    d = tmp_path_factory.mktemp("models")
    paths = [
        train("td3", seed=0, total_steps=500, out_dir=d, log_dir=None, **SMOKE),
        train("td3", seed=1, total_steps=500, out_dir=d, log_dir=None, **SMOKE),
        train("sac", seed=0, total_steps=500, out_dir=d, log_dir=None, **SMOKE),
        train("td3", seed=0, total_steps=500, out_dir=d, log_dir=None, shielded=False, **SMOKE),
    ]
    names = sorted(Path(p).name for p in paths)
    assert names == ["sac_seed0.zip", "td3_noshield_seed0.zip", "td3_seed0.zip", "td3_seed1.zip"]
    assert all(Path(p).exists() for p in paths)
    return d


def test_train_rejects_unknown_algo(tmp_path):
    from train import train
    with pytest.raises(ValueError):
        train("ddpg", seed=0, total_steps=10, out_dir=tmp_path, log_dir=None)


def test_make_env_is_seeded_and_shielded():
    from agents.shield import ShieldedEnv
    from train import make_env
    env = make_env(seed=3, shielded=True)
    assert isinstance(env.unwrapped.__class__.__name__, str)
    found, e = False, env
    while hasattr(e, "env"):
        found |= isinstance(e, ShieldedEnv)
        e = e.env
    assert found
    raw = make_env(seed=3, shielded=False)
    e = raw
    while hasattr(e, "env"):
        assert not isinstance(e, ShieldedEnv)
        e = e.env


def test_trained_model_loads_and_predicts(model_dir):
    from stable_baselines3 import SAC, TD3
    for cls, name in ((TD3, "td3_seed0.zip"), (SAC, "sac_seed0.zip")):
        model = cls.load(model_dir / name, device="cpu")
        a, _ = model.predict(np.zeros(5, dtype=np.float32), deterministic=True)
        assert a.shape == (1,) and -1.0 <= float(a[0]) <= 1.0


def test_tune_and_pareto(tmp_path):
    from tune_baselines import pareto_sweep, tune
    gains = tune(n_trials=4, n_scenarios=2, seed=0, out_path=tmp_path / "gains.json")
    assert set(gains) == {"droop_k", "pi_kp", "pi_ki", "pid_kp", "pid_ki", "pid_kd"}
    assert all(isinstance(v, float) and v >= 0.0 for v in gains.values())
    assert json.loads((tmp_path / "gains.json").read_text(encoding="utf-8")) == gains

    sweep = pareto_sweep(n_scenarios=2, seed=0, out_path=tmp_path / "pareto.json")
    assert len(sweep) >= 6
    assert all({"kp", "ki", "rmse_mhz", "bess_throughput_kwh"} <= set(row) for row in sweep)
    assert json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8")) == sweep


@pytest.fixture(scope="module")
def eval_results(model_dir, tmp_path_factory):
    from evaluate import run_evaluation
    out = tmp_path_factory.mktemp("results")
    figs = tmp_path_factory.mktemp("figures")
    results = run_evaluation(n_scenarios=3, seed=7, model_dir=model_dir, out_dir=out,
                             fig_dir=figs, gains_path=None)
    return results, out, figs


def test_evaluation_structure(eval_results):
    results, out, _ = eval_results
    on_disk = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(results))
    assert {"meta", "controllers", "comparisons", "pareto"} <= set(results)
    assert results["meta"]["n_scenarios"] == 3 and results["meta"]["seed"] == 7

    ctrls = results["controllers"]
    assert set(ctrls) == {CTRL_NOBESS, CTRL_DROOP, CTRL_PI, CTRL_PID,
                          CTRL_TD3_SHIELD, CTRL_SAC_SHIELD, CTRL_TD3_RAW}
    for name, entry in ctrls.items():
        assert {"mean", "std", "ci_low", "ci_high"} <= set(entry["metrics"]["rmse_mhz"]), name
        assert len(entry["per_scenario"]["rmse_mhz"]) == 3, name
        assert "bess_throughput_kwh" in entry["metrics"], name

    assert ctrls[CTRL_NOBESS]["metrics"]["bess_throughput_kwh"]["mean"] == pytest.approx(0.0)
    for name in (CTRL_NOBESS, CTRL_DROOP, CTRL_PI, CTRL_PID, CTRL_TD3_RAW):
        assert ctrls[name]["metrics"]["autonomy_pct"]["mean"] == pytest.approx(100.0), name
    assert set(ctrls[CTRL_TD3_SHIELD]["per_seed"]) == {"0", "1"}
    assert set(ctrls[CTRL_SAC_SHIELD]["per_seed"]) == {"0"}

    comps = results["comparisons"]
    for rl in (CTRL_TD3_SHIELD, CTRL_SAC_SHIELD, CTRL_TD3_RAW):
        for base in (CTRL_PI, CTRL_PID):
            assert 0.0 <= comps[f"{rl}_vs_{base}"]["rmse_mhz"]["p_value"] <= 1.0
    assert results["pareto"]["PI_sweep"] == []     # pareto dosyası verilmedi


def test_evaluation_is_paired(eval_results):
    """Tüm kontrolcüler aynı senaryoları görmeli: NoBESS ve PI aynı bozucuya farklı yanıt verir."""
    results, _, _ = eval_results
    none = results["controllers"][CTRL_NOBESS]["per_scenario"]["rmse_mhz"]
    pi = results["controllers"][CTRL_PI]["per_scenario"]["rmse_mhz"]
    assert all(p < n for p, n in zip(pi, none))


def test_evaluation_figures(eval_results):
    _, _, figs = eval_results
    for name in FIGURES:
        f = figs / name
        assert f.exists() and f.stat().st_size > 5_000, name


def test_evaluation_is_deterministic(eval_results, model_dir, tmp_path):
    from evaluate import run_evaluation
    _, out, _ = eval_results
    run_evaluation(n_scenarios=3, seed=7, model_dir=model_dir, out_dir=tmp_path,
                   fig_dir=None, gains_path=None)
    assert (tmp_path / "results.json").read_bytes() == (out / "results.json").read_bytes()


def test_evaluation_without_models(tmp_path):
    from evaluate import run_evaluation
    empty = tmp_path / "no_models"
    empty.mkdir()
    results = run_evaluation(n_scenarios=2, seed=1, model_dir=empty, out_dir=tmp_path,
                             fig_dir=tmp_path / "figs", gains_path=None)
    assert set(results["controllers"]) == {CTRL_NOBESS, CTRL_DROOP, CTRL_PI, CTRL_PID}
    assert results["comparisons"] == {}
    assert all((tmp_path / "figs" / name).exists() for name in FIGURES)


def test_report_table_and_readme_update(eval_results, tmp_path):
    from utils.report import render_results_table, update_readme
    results, _, _ = eval_results
    table = render_results_table(results)
    assert table == render_results_table(results)
    header = table.splitlines()[0]
    for name in (CTRL_NOBESS, CTRL_PI, CTRL_TD3_SHIELD, CTRL_SAC_SHIELD):
        assert name in header
    assert "±" in table
    rmse = results["controllers"][CTRL_PI]["metrics"]["rmse_mhz"]["mean"]
    assert f"{rmse:.1f}" in table

    readme = tmp_path / "README.md"
    readme.write_text("# Başlık\n\n<!-- RESULTS:START -->\neski\n<!-- RESULTS:END -->\n\nson\n", encoding="utf-8")
    update_readme(results, readme)
    text = readme.read_text(encoding="utf-8")
    assert "eski" not in text and text.startswith("# Başlık") and text.rstrip().endswith("son")
    block = text.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    assert block.strip() == table.strip()
