"""
evaluate.py — TD3 Safe-AI vs Geleneksel PI Karşılaştırması
============================================================
5 bağımsız trial ortalaması kullanılır (tekrarlanabilir sonuçlar).

Üretilen Grafikler:
  Fig 1 — Ana Karşılaştırma : Frekans + Kontrol Eforu
  Fig 2 — Yük Darbesi Zoom  : t=35-60s + toparlanma süresi
  Fig 3 — Frekans Histogramı: Δf dağılımı
  Fig 4 — Kümülatif Enerji  : Biriken kontrol eforu + CO₂ tasarruf alanı
  Fig 5 — Shield Şeridi      : AI/PI aktivasyon + ROCOF
  Fig 6 — Özet Tablo         : Konfor bandı çubuk + performans tablosu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import TD3
from envs.microgrid_lfc_env import MicrogridLFCEnv
from agents.safe_hybrid_controller import SafeHybridController
from utils.Data_loader import load_epias_wind_data


# ══════════════════════════════════════════════════════════════
# METRİK HESAPLAMA
# ══════════════════════════════════════════════════════════════

def calculate_metrics(delta_f_hz, p_ess_real, safety_flags,
                      dt=0.1, base_power_mw=10.0):
    delta_f = np.array(delta_f_hz)
    actions = np.array(p_ess_real)
    N       = len(delta_f)

    rmse    = float(np.sqrt(np.mean(delta_f ** 2)))
    nadir   = float(np.min(delta_f))
    zenith  = float(np.max(delta_f))
    max_dev = float(np.max(np.abs(delta_f)))

    effort_mwh = float(np.sum(np.abs(actions)) * dt * base_power_mw / 3600.0)
    jitter     = float(np.sum(np.abs(np.diff(actions))))

    ai_steps     = int(np.sum(safety_flags))
    autonomy_pct = (ai_steps / max(N, 1)) * 100.0
    violations   = int(np.sum(np.abs(delta_f) > 0.150))

    band_50  = float(np.sum(np.abs(delta_f) < 0.050) / N * 100)
    band_100 = float(np.sum(np.abs(delta_f) < 0.100) / N * 100)
    band_150 = float(np.sum(np.abs(delta_f) < 0.150) / N * 100)

    rocof     = np.abs(np.diff(delta_f)) / dt
    rocof_max = float(np.max(rocof))
    rocof_avg = float(np.mean(rocof))

    # Toparlanma: darbe sonrası 15 adım bekle, sonra ±20mHz bandına dönüşü ölç
    SKIP = 15
    post = delta_f[400 + SKIP:]
    st   = next((i for i, v in enumerate(post) if abs(v) < 0.020), None)
    settling_time = float((st + SKIP) * dt) if st is not None else float("nan")

    return {
        "rmse": rmse, "nadir": nadir, "zenith": zenith, "max_dev": max_dev,
        "effort_mwh": effort_mwh, "jitter": jitter, "autonomy": autonomy_pct,
        "violations": violations, "band_50": band_50, "band_100": band_100,
        "band_150": band_150, "rocof_max": rocof_max, "rocof_avg": rocof_avg,
        "settling_time": settling_time,
    }


# ══════════════════════════════════════════════════════════════
# SENARYO ÇALIŞTIRICI
# ══════════════════════════════════════════════════════════════

def run_scenario(env, controller, is_hybrid, label, unseen_data,
                 impulse_step=400, impulse_size=0.2, env_seed=2024):
    obs, _ = env.reset(seed=env_seed)
    env.wind_profile = unseen_data

    delta_f_list, p_ess_list, shield_list = [], [], []
    integral_pi = 0.0

    for step in range(1000):
        if step == impulse_step:
            env.base_load += impulse_size

        if is_hybrid:
            action, mode = controller.predict(obs, deterministic=True)
            shield_list.append(1 if mode == "AI" else 0)
        else:
            delta_f_hz  = float(obs[0]) * 0.3
            error       = -delta_f_hz
            integral_pi += error * 0.1
            integral_pi  = np.clip(integral_pi, -0.5, 0.5)
            action = np.array([float(np.clip(
                2.0 * error + 0.5 * integral_pi, -1.0, 1.0
            ))], dtype=np.float32)
            shield_list.append(0)

        obs, _, terminated, truncated, info = env.step(action)
        delta_f_list.append(info["delta_f"])
        p_ess_list.append(info["p_ess"])

        if terminated or truncated:
            break

    print(f"    [{label}] {len(delta_f_list)} adim tamamlandi.")
    return np.array(delta_f_list), np.array(p_ess_list), np.array(shield_list)


# ══════════════════════════════════════════════════════════════
# ANA DEĞERLENDİRME
# ══════════════════════════════════════════════════════════════

def run_evaluation(n_trials=15):
    print("\n" + "="*65)
    print(f"  TD3 Safe-AI vs Geleneksel PI — {n_trials} Trial Ortalamasi")
    print("="*65 + "\n")

    # Veri
    csv_path = "data/Gercek_Zamanli_Uretim-20112025-20022026.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/epias_wind_data.csv"

    full_data        = load_epias_wind_data(filepath=csv_path, max_steps=2000)
    unseen_test_data = full_data[1000:2000].astype(np.float32)

    # Model
    print("Model yukleniyor...")
    custom_objs = {"learning_rate": lambda _: 3e-4, "lr_schedule": lambda _: 3e-4}
    try:
        td3_model = TD3.load("models/td3_microgrid_model", custom_objects=custom_objs)
    except Exception:
        td3_model = TD3.load("models/td3_microgrid_model.zip", custom_objects=custom_objs)
    print("  [OK] TD3 modeli yuklendi.\n")

    # 5 Trial döngüsü
    SEEDS = list(range(2024, 2024 + n_trials))
    all_df_pi,  all_pe_pi,  all_sh_pi  = [], [], []
    all_df_td3, all_pe_td3, all_sh_td3 = [], [], []

    for i, seed in enumerate(SEEDS):
        print(f"  Trial {i+1}/{n_trials}  (seed={seed})")
        np.random.seed(seed)

        ctrl_td3 = SafeHybridController(td3_model, Kp=2.0, Ki=0.5, safety_limit=0.15)
        ctrl_td3.reset()

        env_pi_t  = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
        env_td3_t = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)

        df_pi_t,  pe_pi_t,  sh_pi_t  = run_scenario(env_pi_t,  None,     False, "PI",  unseen_test_data, env_seed=seed)
        df_td3_t, pe_td3_t, sh_td3_t = run_scenario(env_td3_t, ctrl_td3, True,  "TD3", unseen_test_data, env_seed=seed)

        all_df_pi.append(df_pi_t);   all_pe_pi.append(pe_pi_t);   all_sh_pi.append(sh_pi_t)
        all_df_td3.append(df_td3_t); all_pe_td3.append(pe_td3_t); all_sh_td3.append(sh_td3_t)

    # Grafik için seed=2024 verisi
    df_pi,  pe_pi,  sh_pi  = all_df_pi[0],  all_pe_pi[0],  all_sh_pi[0]
    df_td3, pe_td3, sh_td3 = all_df_td3[0], all_pe_td3[0], all_sh_td3[0]

    # Son trial'dan controller log'u al
    ctrl_td3_last = SafeHybridController(td3_model, Kp=2.0, Ki=0.5, safety_limit=0.15)
    ctrl_td3_last.reset()
    env_log = MicrogridLFCEnv(data_path=csv_path, max_steps=1000)
    np.random.seed(2024)
    run_scenario(env_log, ctrl_td3_last, True, "TD3-log", unseen_test_data, env_seed=2024)

    # Ortalama metrikler
    BASE_MW    = 10.0
    CO2_FACTOR = 400.0

    def avg_metrics(all_df, all_pe, all_sh):
        all_m = [calculate_metrics(df, pe, sh, base_power_mw=BASE_MW)
                 for df, pe, sh in zip(all_df, all_pe, all_sh)]
        keys  = all_m[0].keys()
        return (
            {k: float(np.mean([m[k] for m in all_m])) for k in keys},
            {k: float(np.std( [m[k] for m in all_m])) for k in keys},
        )

    m_pi,  s_pi  = avg_metrics(all_df_pi,  all_pe_pi,  all_sh_pi)
    m_td3, s_td3 = avg_metrics(all_df_td3, all_pe_td3, all_sh_td3)

    sav_mwh  = m_pi["effort_mwh"] - m_td3["effort_mwh"]
    sav_pct  = sav_mwh / max(m_pi["effort_mwh"], 1e-9) * 100
    co2_sav  = sav_mwh * CO2_FACTOR

    # ── Konsol Raporu ─────────────────────────────────────────
    W = 22
    print(f"\n\n{'='*75}")
    print(f"  {n_trials} TRIAL ORTALAMASI — PERFORMANS ANALİZİ  (Ort ± Std)")
    print(f"{'='*75}")
    print(f"  {'Metrik':<33} | {'Geleneksel PI':>{W}} | {'TD3 Safe-AI (Onerilen)':>{W}}")
    print(f"  {'-'*73}")

    rows = [
        ("Frekans RMSE (mHz)",       m_pi["rmse"]*1e3,       s_pi["rmse"]*1e3,       m_td3["rmse"]*1e3,       s_td3["rmse"]*1e3,       ".2f"),
        ("Maksimum Sapma (mHz)",      m_pi["max_dev"]*1e3,    s_pi["max_dev"]*1e3,    m_td3["max_dev"]*1e3,    s_td3["max_dev"]*1e3,    ".2f"),
        ("Frekans Nadiri (mHz)",      m_pi["nadir"]*1e3,      s_pi["nadir"]*1e3,      m_td3["nadir"]*1e3,      s_td3["nadir"]*1e3,      ".2f"),
        ("Kontrol Eforu (MWh)",       m_pi["effort_mwh"],     s_pi["effort_mwh"],     m_td3["effort_mwh"],     s_td3["effort_mwh"],     ".5f"),
        ("Sinyal Jitter",             m_pi["jitter"],         s_pi["jitter"],         m_td3["jitter"],         s_td3["jitter"],         ".2f"),
        ("Otonomi Orani (%)",         m_pi["autonomy"],       s_pi["autonomy"],       m_td3["autonomy"],       s_td3["autonomy"],       ".1f"),
        ("Guvenlik Ihlali",           float(m_pi["violations"]), float(s_pi["violations"]), float(m_td3["violations"]), float(s_td3["violations"]), ".1f"),
        ("Bant |Df|<50mHz (%)",       m_pi["band_50"],        s_pi["band_50"],        m_td3["band_50"],        s_td3["band_50"],        ".1f"),
        ("Bant |Df|<100mHz (%)",      m_pi["band_100"],       s_pi["band_100"],       m_td3["band_100"],       s_td3["band_100"],       ".1f"),
        ("Bant |Df|<150mHz (%)",      m_pi["band_150"],       s_pi["band_150"],       m_td3["band_150"],       s_td3["band_150"],       ".1f"),
        ("ROCOF Maksimum (Hz/s)",     m_pi["rocof_max"],      s_pi["rocof_max"],      m_td3["rocof_max"],      s_td3["rocof_max"],      ".4f"),
        ("ROCOF Ortalama (Hz/s)",     m_pi["rocof_avg"],      s_pi["rocof_avg"],      m_td3["rocof_avg"],      s_td3["rocof_avg"],      ".4f"),
        ("Toparlanma Suresi (s)",     m_pi["settling_time"],  s_pi["settling_time"],  m_td3["settling_time"],  s_td3["settling_time"],  ".1f"),
    ]
    for name, vpi, spi, vtd3, std3, fmt in rows:
        pi_str  = f"{vpi:{fmt}} ± {spi:{fmt}}"
        td3_str = f"{vtd3:{fmt}} ± {std3:{fmt}}"
        print(f"  {name:<33} | {pi_str:>{W}} | {td3_str:>{W}}")

    print(f"  {'-'*73}")
    print(f"  Enerji Tasarrufu        : %{sav_pct:.2f}  ({sav_mwh:.5f} MWh/ep)")
    print(f"  Engellenen CO2          : {co2_sav:.4f} kg")
    print(f"  AI / PI adim (seed=2024): {ctrl_td3_last.usage_log['AI']} / {ctrl_td3_last.usage_log['PI']}")
    print(f"{'='*75}\n")

    # ══════════════════════════════════════════════════════════
    # GRAFİKLER
    # ══════════════════════════════════════════════════════════
    os.makedirs("figures", exist_ok=True)
    C_PI  = "#d62728"
    C_TD3 = "#1f77b4"
    C_BND = "#2ca02c"

    t    = np.arange(len(df_td3)) * 0.1
    t_pi = np.arange(len(df_pi))  * 0.1

    # ── Fig 1: Ana Karşılaştırma ─────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig1.suptitle("Sistem Kararlılık Analizi: TD3 Safe-AI vs Geleneksel PI",
                  fontsize=13, fontweight="bold")

    ax1.plot(t_pi, df_pi,  color=C_PI,  linestyle="--", alpha=0.75, linewidth=1.8,
             label="Geleneksel PI")
    ax1.plot(t,    df_td3, color=C_TD3, linewidth=2.2,
             label="TD3 Safe-AI (Onerilen)")
    ax1.axhline( 0.15, color="black", linestyle=":", alpha=0.35, label="Guvenlik Siniri ±150mHz")
    ax1.axhline(-0.15, color="black", linestyle=":", alpha=0.35)
    ax1.axvline(40, color="orange", linestyle="-.", alpha=0.8, linewidth=1.5,
                label="Yuk Darbesi +0.2 pu (t=40s)")
    ax1.fill_between(t_pi, -0.05, 0.05, alpha=0.08, color=C_BND, label="Konfor Bandi ±50mHz")
    ax1.set_ylabel("Frekans Sapmasi (Hz)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Frekans Sapmasi Karsilastirmasi")

    ax2.plot(t_pi, pe_pi,  color=C_PI,  linestyle="--", alpha=0.7, linewidth=1.8, label="PI")
    ax2.plot(t,    pe_td3, color=C_TD3, linewidth=2.0, label="TD3 Safe-AI")
    ax2.set_ylabel("Batarya Gucu (pu)", fontsize=11)
    ax2.set_xlabel("Zaman (s)", fontsize=11)
    ax2.set_title(f"Kontrol Eforu  |  TD3 Tasarrufu: %{sav_pct:.1f}  |  CO₂: {co2_sav:.2f} kg")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("figures/Fig1_Ana_Karsilastirma.png", dpi=300, bbox_inches="tight")
    print("[Grafik 1] Fig1_Ana_Karsilastirma.png")
    plt.close()

    # ── Fig 2: Yük Darbesi Zoom ──────────────────────────────
    z0, z1 = 350, 600
    tz = np.arange(z1 - z0) * 0.1 + z0 * 0.1

    fig2, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tz, df_pi [z0:z1], color=C_PI,  linestyle="--", alpha=0.8, linewidth=1.8, label="PI")
    ax.plot(tz, df_td3[z0:z1], color=C_TD3, linewidth=2.2, label="TD3 Safe-AI (Onerilen)")
    ax.axhline( 0.15, color="black", linestyle=":", alpha=0.4, label="Guvenlik Siniri ±150mHz")
    ax.axhline(-0.15, color="black", linestyle=":", alpha=0.4)
    ax.axhline( 0.02, color=C_BND,  linestyle=":", alpha=0.6, label="Toparlanma Bandi ±20mHz")
    ax.axhline(-0.02, color=C_BND,  linestyle=":", alpha=0.6)
    ax.axvline(40, color="orange", linestyle="-.", linewidth=1.5, alpha=0.85,
               label="Yuk Darbesi (t=40s)")

    SKIP = 15
    for lbl_s, df_arr, color, y_off in [("PI", df_pi, C_PI, 0.008), ("TD3", df_td3, C_TD3, 0.018)]:
        post = df_arr[400 + SKIP:]
        st   = next((i for i, v in enumerate(post) if abs(v) < 0.020), None)
        if st is not None:
            total_steps = st + SKIP
            tx = 40 + total_steps * 0.1
            ty = df_arr[min(400 + total_steps, len(df_arr) - 1)]
            if z0 * 0.1 <= tx <= z1 * 0.1:
                ax.annotate(f"{lbl_s}: {total_steps*0.1:.1f}s",
                            xy=(tx, ty), xytext=(tx + 1.5, ty + y_off),
                            fontsize=9, color=color, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.3))
            else:
                idx = ["PI", "TD3"].index(lbl_s)
                ax.text(0.98, 0.97 - idx * 0.08,
                        f"{lbl_s} toparlanma: {total_steps*0.1:.1f}s",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Zaman (s)", fontsize=11)
    ax.set_ylabel("Frekans Sapmasi (Hz)", fontsize=11)
    ax.set_title("Yuk Darbesi Toparlanma Analizi (t = 35 – 60 s)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("figures/Fig2_Yuk_Darbesi_Zoom.png", dpi=300, bbox_inches="tight")
    print("[Grafik 2] Fig2_Yuk_Darbesi_Zoom.png")
    plt.close()

    # ── Fig 3: Histogram ─────────────────────────────────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig3.suptitle(f"Frekans Sapmasi Dagilimi ({n_trials} Trial Ortalamasi)", fontsize=12, fontweight="bold")
    bins = np.linspace(-0.16, 0.16, 50)

    for ax_h, data, color, lbl, m in [
        (ax3a, df_pi,  C_PI,  "Geleneksel PI",        m_pi),
        (ax3b, df_td3, C_TD3, "TD3 Safe-AI (Onerilen)", m_td3),
    ]:
        ax_h.hist(data * 1000, bins=bins * 1000, color=color, alpha=0.75,
                  edgecolor="white", linewidth=0.4)
        ax_h.axvline( 150, color="black", linestyle="--", alpha=0.5, label="±150mHz siniri")
        ax_h.axvline(-150, color="black", linestyle="--", alpha=0.5)
        ax_h.axvline(   0, color="gray",  linestyle=":",  alpha=0.5)
        ax_h.set_title(
            f"{lbl}\n"
            f"RMSE={m['rmse']*1000:.1f} mHz  |  "
            f"Bant<100mHz: {m['band_100']:.0f}%  |  "
            f"Ihlal: {m['violations']:.0f}",
            fontsize=10
        )
        ax_h.set_xlabel("Frekans Sapmasi (mHz)")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, alpha=0.2)

    ax3a.set_ylabel("Adim Sayisi")
    plt.tight_layout()
    plt.savefig("figures/Fig3_Frekans_Histogram.png", dpi=300, bbox_inches="tight")
    print("[Grafik 3] Fig3_Frekans_Histogram.png")
    plt.close()

    # ── Fig 4: Kümülatif Enerji ──────────────────────────────
    cum_pi  = np.cumsum(np.abs(pe_pi))  * 0.1 * BASE_MW / 3600 * 1000
    cum_td3 = np.cumsum(np.abs(pe_td3)) * 0.1 * BASE_MW / 3600 * 1000

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    ax4.plot(t_pi, cum_pi,  color=C_PI,  linestyle="--", alpha=0.8, linewidth=1.8, label="PI")
    ax4.plot(t,    cum_td3, color=C_TD3, linewidth=2.2, label="TD3 Safe-AI (Onerilen)")
    ax4.axvline(40, color="orange", linestyle="-.", alpha=0.8, linewidth=1.5,
                label="Yuk Darbesi")
    min_len = min(len(cum_pi), len(cum_td3))
    ax4.fill_between(t[:min_len], cum_td3[:min_len], cum_pi[:min_len],
                     alpha=0.15, color=C_TD3,
                     label=f"Tasarruf Alani  %{sav_pct:.1f}  ({co2_sav:.2f} kg CO₂)")
    ax4.set_xlabel("Zaman (s)", fontsize=11)
    ax4.set_ylabel("Kumulatif Kontrol Eforu (Wh × 10⁻³)", fontsize=11)
    ax4.set_title("Kumulatif Enerji Tuketimi ve CO₂ Tasarrufu", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("figures/Fig4_Kumulatif_Enerji.png", dpi=300, bbox_inches="tight")
    print("[Grafik 4] Fig4_Kumulatif_Enerji.png")
    plt.close()

    # ── Fig 5: Shield Şeridi + ROCOF ─────────────────────────
    fig5, (ax5a, ax5b, ax5c) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                              gridspec_kw={"height_ratios": [3, 1, 2]})
    fig5.suptitle("Guvenlik Kalkani Aktivasyonu ve ROCOF Analizi", fontsize=13, fontweight="bold")

    ax5a.plot(t_pi, df_pi,  color=C_PI,  linestyle="--", alpha=0.7, linewidth=1.8, label="PI")
    ax5a.plot(t,    df_td3, color=C_TD3, linewidth=2.2, label="TD3 Safe-AI")
    ax5a.axhline( 0.15, color="black", linestyle=":", alpha=0.4, label="Guvenlik Siniri")
    ax5a.axhline(-0.15, color="black", linestyle=":", alpha=0.4)
    ax5a.axvline(40, color="orange", linestyle="-.", alpha=0.7)
    ax5a.set_ylabel("Frekans Sapmasi (Hz)")
    ax5a.legend(fontsize=9, loc="upper right")
    ax5a.grid(True, alpha=0.2)

    for i in range(min(len(sh_td3), len(t) - 1)):
        color = C_TD3 if sh_td3[i] == 1 else C_PI
        ax5b.axvspan(t[i], t[i] + 0.1, color=color, alpha=0.65, linewidth=0)
    ai_p = mpatches.Patch(color=C_TD3, alpha=0.65,
                           label=f"AI Aktif ({ctrl_td3_last.usage_log['AI']} adim  —  %{m_td3['autonomy']:.1f})")
    pi_p = mpatches.Patch(color=C_PI,  alpha=0.65,
                           label=f"PI Aktif ({ctrl_td3_last.usage_log['PI']} adim)")
    ax5b.legend(handles=[ai_p, pi_p], fontsize=8, loc="upper right")
    ax5b.set_yticks([])
    ax5b.set_ylabel("Mod", fontsize=9)
    ax5b.set_title("Guvenlik Kalkani Aktivasyon Seridi (seed=2024)", fontsize=10)

    rocof_pi_arr  = np.abs(np.diff(df_pi))  / 0.1
    rocof_td3_arr = np.abs(np.diff(df_td3)) / 0.1
    ax5c.plot(t_pi[:-1], rocof_pi_arr,  color=C_PI,  linestyle="--", alpha=0.7, linewidth=1.5,
              label=f"PI   ort={rocof_pi_arr.mean():.3f} Hz/s  maks={rocof_pi_arr.max():.3f} Hz/s")
    ax5c.plot(t[:-1],    rocof_td3_arr, color=C_TD3, linewidth=1.8,
              label=f"TD3  ort={rocof_td3_arr.mean():.3f} Hz/s  maks={rocof_td3_arr.max():.3f} Hz/s")
    ax5c.set_ylabel("ROCOF (Hz/s)")
    ax5c.set_xlabel("Zaman (s)")
    ax5c.legend(fontsize=8)
    ax5c.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("figures/Fig5_Shield_ROCOF.png", dpi=300, bbox_inches="tight")
    print("[Grafik 5] Fig5_Shield_ROCOF.png")
    plt.close()

    # ── Fig 6: Konfor Bandı + Özet Tablo ─────────────────────
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 5))
    fig6.suptitle("Performans Ozeti", fontsize=13, fontweight="bold")

    bands  = ["|Δf|<50mHz", "|Δf|<100mHz", "|Δf|<150mHz"]
    pi_b   = [m_pi["band_50"],  m_pi["band_100"],  m_pi["band_150"]]
    td3_b  = [m_td3["band_50"], m_td3["band_100"], m_td3["band_150"]]
    x, w   = np.arange(3), 0.32

    ax6a.bar(x - w/2, pi_b,  w, label="Geleneksel PI",        color=C_PI,  alpha=0.75)
    ax6a.bar(x + w/2, td3_b, w, label="TD3 Safe-AI (Onerilen)", color=C_TD3, alpha=0.75)
    ax6a.set_xticks(x)
    ax6a.set_xticklabels(bands, fontsize=10)
    ax6a.set_ylabel("Zaman Yuzdesi (%)")
    ax6a.set_ylim(0, 110)
    ax6a.set_title(f"Konfor Bandinda Gecirilen Sure ({n_trials} Trial Ort.)")
    ax6a.legend(fontsize=9)
    ax6a.grid(True, axis="y", alpha=0.3)
    for bars in ax6a.containers:
        ax6a.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)

    ax6b.axis("off")
    col_labels = ["Metrik", "Geleneksel PI (Ort ± Std)", "TD3 Safe-AI (Ort ± Std)"]
    cell_data  = [
        ["RMSE (mHz)",           f"{m_pi['rmse']*1e3:.2f} ± {s_pi['rmse']*1e3:.2f}",        f"{m_td3['rmse']*1e3:.2f} ± {s_td3['rmse']*1e3:.2f}"],
        ["Maks Sapma (mHz)",     f"{m_pi['max_dev']*1e3:.2f} ± {s_pi['max_dev']*1e3:.2f}",  f"{m_td3['max_dev']*1e3:.2f} ± {s_td3['max_dev']*1e3:.2f}"],
        ["Kontrol Eforu (MWh)",  f"{m_pi['effort_mwh']:.5f} ± {s_pi['effort_mwh']:.5f}",    f"{m_td3['effort_mwh']:.5f} ± {s_td3['effort_mwh']:.5f}"],
        ["Enerji Tasarrufu",     "—",                                                          f"%{sav_pct:.1f}"],
        ["CO₂ Tasarrufu (kg)",   "—",                                                          f"{co2_sav:.2f}"],
        ["Otonomi (%)",          f"0.0",                                                       f"{m_td3['autonomy']:.1f} ± {s_td3['autonomy']:.1f}"],
        ["Guvenlik Ihlali",      f"{m_pi['violations']:.1f} ± {s_pi['violations']:.1f}",      f"{m_td3['violations']:.1f} ± {s_td3['violations']:.1f}"],
        ["ROCOF Ort. (Hz/s)",    f"{m_pi['rocof_avg']:.4f} ± {s_pi['rocof_avg']:.4f}",       f"{m_td3['rocof_avg']:.4f} ± {s_td3['rocof_avg']:.4f}"],
        ["Toparlanma (s)",       f"{m_pi['settling_time']:.1f} ± {s_pi['settling_time']:.1f}",f"{m_td3['settling_time']:.1f} ± {s_td3['settling_time']:.1f}"],
    ]
    tbl = ax6b.table(cellText=cell_data, colLabels=col_labels,
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_data) + 1):
        tbl[i, 2].set_facecolor("#d6eaf8")
    ax6b.set_title(f"Ozet Performans Tablosu ({n_trials} Trial — Ort ± Std)", fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig("figures/Fig6_Ozet_Tablo.png", dpi=300, bbox_inches="tight")
    print("[Grafik 6] Fig6_Ozet_Tablo.png")
    plt.close()

    print(f"\n Tum grafikler 'figures/' klasorune kaydedildi.")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_evaluation(n_trials=15)