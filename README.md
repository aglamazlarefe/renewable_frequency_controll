# ⚡ Renewable Frequency Control via TD3 + Safe-AI Hybrid

**Deep Reinforcement Learning for Secondary Frequency Control in Renewable-Dominated Microgrids**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-2.0+-green)](https://github.com/DLR-RM/stable-baselines3)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-orange)](https://gymnasium.farama.org/)

---

## 📖 Overview

This project presents a **Safe-AI hybrid control framework** for secondary frequency regulation in renewable-rich microgrids. A **TD3 (Twin Delayed DDPG)** agent manages the Battery Energy Storage System (BESS) under normal conditions, while a classical **PI controller** acts as a safety shield when frequency deviations exceed ±150 mHz — ensuring both **energy efficiency** and **grid reliability**.

| Key Achievement | Value |
|:---|---:|
| Energy Savings vs Pure PI | **21.5%** |
| CO₂ Avoided per Episode | **5.6 kg** |
| AI Autonomy Rate | **93.4%** |
| Training Time (RTX 3060) | ~8 minutes |

---

## 🧠 The Problem

Modern power grids face a critical challenge: **renewable energy sources** (wind, solar) are inherently intermittent. Unlike synchronous generators, they provide **no rotating inertia**, making frequency regulation significantly harder. Traditional PI/PID controllers respond reliably but consume excessive battery energy, while pure RL agents can behave unpredictably near safety boundaries.

**This project's insight**: Combine the *efficiency* of Deep RL with the *safety guarantees* of classical control.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────┐
                    │   EPİAŞ Wind Data (CSV)  │
                    │   (2226 hours, Turkey)   │
                    └────────┬────────────────┘
                             │
                    ┌────────▼───────────────┐
                    │   Data Loader           │
                    │   (normalize + interp)  │
                    └────────┬────────────────┘
                             │
                    ┌────────▼───────────────┐
                    │   MicrogridLFCEnv       │
                    │   (Gymnasium)           │
                    │                         │
                    │   • Swing Equation       │
                    │   • Virtual Inertia      │
                    │   • Safety Shield (PI)   │
                    │   • Slew Rate Limit      │
                    └────┬──────────┬─────────┘
                         │          │
              ┌──────────▼──┐  ┌────▼──────────┐
              │  train.py   │  │  evaluate.py   │
              │  (TD3 SB3)  │  │ (15 Trials)    │
              │  ~8 min     │  │ 6 Figures      │
              └─────────────┘  └────┬──────────┘
                                    │
              ┌─────────────────────▼──────────┐
              │   SafeHybridController          │
              │   ┌────────┐  ┌────────┐        │
              │   │ TD3 AI │  │ PI Shield │       │
              │   │ (93%)  │  │  (7%)   │       │
              │   └────────┘  └────────┘        │
              │   ⚡ Bumpless Transfer           │
              └──────────────────────────────────┘
```

### 🔄 How Safe-AI Works

1. **Normal Operation** → TD3 agent controls the battery (energy-efficient, low wear)
2. **|Δf| > 150 mHz** → PI controller takes over (guaranteed recovery)
3. **Bumpless Transfer** → integral term decays smoothly during AI → PI transitions
4. **Dual protection** → PI shield exists both inside the environment AND in the controller

---

## 📊 Results (15 Independent Trials)

| Metric | Pure PI | TD3 Safe-AI (Ours) |
|:---|---:|---:|
| Freq RMSE (mHz) | 51.13 ± 2.5 | 61.49 ± 5.8 |
| Max Deviation (mHz) | 112.46 ± 2.8 | 133.97 ± 12.1 |
| Control Effort (MWh) | 0.06498 ± 0.002 | **0.05099 ± 0.003** |
| Signal Jitter | 9.19 ± 0.3 | 20.53 ± 4.0 |
| Autonomy (%) | 0.0 | **93.4 ± 2.1** |
| Band <50 mHz (%) | 54.7 | 40.2 |
| Band <100 mHz (%) | 97.6 | 78.5 |
| Band <150 mHz (%) | 100.0 | 91.9 |
| ROCOF Mean (Hz/s) | 0.0075 | 0.0104 |
| Settling Time (s) | 4.7 | 7.2 |

**The trade-off**: Safe-AI saves **21.5% energy** and prevents **5.6 kg CO₂/episode** while maintaining 93% autonomy — at the cost of slightly higher frequency deviation and jitter during RL exploration.

### 📈 Generated Figures

| Figure | Description |
|:---|---:|
| Fig1 — Main Comparison | Frequency & control effort time series |
| Fig2 — Load Disturbance Zoom | 35–60s window with settling time annotations |
| Fig3 — Frequency Histogram | Δf distribution with safety bands |
| Fig4 — Cumulative Energy | Energy savings area + CO₂ annotation |
| Fig5 — Shield Activation Strip | AI/PI mode timeline + ROCOF |
| Fig6 — Summary Table | All metrics with bar chart |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA (optional, CPU fallback available)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/renewable-frequency-control.git
cd renewable-frequency-control
pip install -r requirements.txt
```

### Training

```bash
# Quick training: 100k steps, ~8 min on RTX 3060
python train.py
```

The model is saved to `models/td3_microgrid_model.zip`. Training logs are written to `logs/` (TensorBoard).

### Evaluation

```bash
# 15-trial comparison with 6 publication-quality figures
python evaluate.py
```

Results appear in `figures/` (6 PNGs) and the console prints the metric table.

---

## 📦 Dependencies

- `numpy` — numerical computing
- `pandas` — data loading
- `matplotlib` — visualization
- `gymnasium` — RL environment standard
- `stable-baselines3` — TD3 implementation
- `torch` — neural network backend

---

## 📁 Repository Structure

```
renewable-frequency-control/
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── .gitignore                 ← Ignored files
├── train.py                   ← TD3 training script
├── evaluate.py                ← 15-trial benchmark + 6 figures
├── envs/
│   ├── __init__.py
│   └── microgrid_lfc_env.py   ← Gymnasium environment (swing equation + safety shield)
├── agents/
│   ├── __init__.py
│   └── safe_hybrid_controller.py ← AI ↔ PI switching logic
├── utils/
│   ├── __init__.py
│   └── Data_loader.py         ← EPİAŞ CSV loader + normalization
└── data/
    └── epias_wind_data.csv    ← Real wind generation data (Turkey EPİAŞ)
```

---

## 📝 Environment Details

| Parameter | Value | Description |
|:---|---:|:---:|
| `M` | 4.0 | Generator inertia constant |
| `D` | 0.8 | Damping coefficient |
| `K_vir` | 0.5 | Virtual inertia gain |
| `dt` | 0.1 s | Simulation time step |
| `Slew rate` | ±0.025 pu | Battery ramp limit per step |
| `Shield threshold` | ±150 mHz | PI takeover boundary |
| `Shield Kp` | 20.0 | Proportional gain |
| `Shield Ki` | 8.0 | Integral gain |
| `Observation` | 3-dim | [Δf/0.2, p_wind, prev_p_ess] |
| `Action` | 1-dim | Battery power command (pu) |

### Reward Function

```
R = +5 (autonomy bonus)  −  120·Δf² (frequency penalty)
    − 5·p² (energy penalty)  −  350·Δp² (jitter penalty)
    − 400·max(0, |Δf|−0.11)² (barrier)
```

When the shield is active: the autonomy bonus becomes −2, and jitter penalty is waived.

---

## 🌱 Future Work

- **Solar + wind hybrid** generation profiles
- **Multi-agent** coordination (multiple BESS units)
- **Transfer learning** across different grid topologies
- **Real-time hardware-in-the-loop** validation
- **SAC vs TD3** architecture comparison
- **Hyperparameter auto-tuning** with Optuna

---

## 📚 References

- Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML.
- EPİAŞ — Turkish Electricity Market Data: [epias.com.tr](https://www.epias.com.tr)
- Stable-Baselines3: [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- Gymnasium: [gymnasium.farama.org](https://gymnasium.farama.org/)

---

## 📄 License

This project is provided for academic and educational purposes.

---

**Author**: Efe Ağlamazlar  
**Contact**: [GitHub Profile](https://github.com/YOUR_USERNAME)