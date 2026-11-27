# SPEC — Model, arayüzler ve tanımlar

Bu belge **sözleşmedir**. Modüller birbirini yalnızca burada yazan arayüzlerle kullanır.
Sayısal değerlerin tek kaynağı `config.py`'dir; burada yazan sayılar yalnızca açıklama içindir.

## 1. Kapsam ve varsayımlar

İzole bir mikro şebeke: dizel jeneratör (birincil droop + yavaş AGC), rüzgâr santrali (bozucu),
yük (bozucu) ve bir BESS. **Kontrol edilen tek büyüklük BESS güç komutudur.**
Küçük sinyal modeli: bütün değişkenler bölüm başındaki denge noktasından sapmadır.

- Güç: sistem tabanında per-unit (`GridParams.s_base_mw`).
- Frekans: dinamik modelde per-unit (`Δf_pu = Δf_Hz / f0`), dış arayüzde **her zaman Hz**.
- İşaretler: `P_ess > 0` deşarj; `ΔP_load > 0` yük artışı; `ΔP_wind > 0` üretim artışı.

EPİAŞ verisi saatliktir; saniye ölçekli frekans dinamiği için yalnızca **çalışma noktasını**
(rüzgâr seviyesi) belirler. Saniyelik değişim sentetiktir (§5). Bu bir sınırlılıktır ve README'de belirtilir.

## 2. Sürekli zaman modeli

Durum vektörü (sıra sabittir): `x = [Δf_pu, ΔP_g, ΔP_m, ΔP_c, P_ess]`
Giriş vektörü (sıra sabittir): `v = [u, d]` — `u`: BESS güç komutu [pu], `d = ΔP_wind − ΔP_load` [pu]

```
dΔf/dt   = (ΔP_m + P_ess + d − D·Δf) / (2H)
dΔP_g/dt = (−ΔP_g − Δf/R + ΔP_c) / T_g           # governor
dΔP_m/dt = (−ΔP_m + ΔP_g) / T_t                   # türbin
dΔP_c/dt = −K_i_agc · (β·Δf − P_ess)              # dizel AGC, β = D + 1/R
dP_ess/dt = (−P_ess + u) / T_ess                  # dönüştürücü gecikmesi
```

AGC'deki `− P_ess` terimi bataryayı boşaltır: kalıcı durumda `β·Δf = P_ess`; frekansı sıfıra
getiren her kontrolcüde batarya gücü de sıfıra döner (zaman sabiti ≈ `1/K_i_agc`).

**Ayrıklaştırma:** kontrol adımı `dt`; doğrusal model sıfırıncı mertebe tutucu (ZOH) ile **kesin**
ayrıklaştırılır (`scipy.linalg.expm`, artırılmış matris yöntemi). Açık Euler kullanılmaz.

**Doğrusal olmayan kısım (ortamda, her adımda, bu sırayla):**
1. `a = clip(action, −1, 1)`, hedef `u* = a · p_max`
2. Hız sınırı: `u = u_prev + clip(u* − u_prev, −ramp_rate·dt, +ramp_rate·dt)`
3. SOC sınırı: `soc ≤ soc_min` ise `u = min(u, 0)`; `soc ≥ soc_max` ise `u = max(u, 0)`
4. `x ← Ad·x + Bd·[u, d_k]`
5. SOC: `e = P_ess · s_base_mw · dt / 3600` [MWh] (adım sonu `P_ess` ile);
   deşarjda `soc −= e / (eta · e_mwh)`, şarjda `soc −= e · eta / e_mwh`; `[0, 1]`'e kırpılır.

## 3. Modül arayüzleri

### 3.1 `utils/data_loader.py`
```python
def load_wind_levels(path: str | Path = DEFAULT_DATA_PATH) -> np.ndarray
    # Saatlik 'Rüzgar' sütunu, en büyük değere bölünmüş, float64, [0, 1].
    # CSV: sep=';', decimal=',', thousands='.'
    # Dosya yoksa FileNotFoundError; sütun yoksa veya veri boşsa ValueError. Hata YUTULMAZ.
def split_wind_levels(levels: np.ndarray, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]
    # Kronolojik: ilk int(train_frac·n) eğitim, kalanı test.
```

### 3.2 `envs/dynamics.py`
```python
STATE_NAMES = ("delta_f_pu", "dp_g", "dp_m", "dp_c", "p_ess")
def build_continuous_model(grid: GridParams, bess: BessParams) -> tuple[np.ndarray, np.ndarray]  # A (5x5), B (5x2)
def build_discrete_model(grid: GridParams, bess: BessParams) -> tuple[np.ndarray, np.ndarray]    # Ad (5x5), Bd (5x2), ZOH @ grid.dt
```

### 3.3 `envs/scenarios.py`
```python
class ScenarioGenerator:
    def __init__(self, wind_levels: np.ndarray,
                 params: ScenarioParams = ScenarioParams(), grid: GridParams = GridParams()): ...
    wind_levels: np.ndarray                                    # kurucuya verilen dizi (açık öznitelik)
    def sample(self, rng: np.random.Generator) -> Scenario     # yalnızca verilen rng'yi kullanır
def make_scenario_bank(wind_levels: np.ndarray, n: int, seed: int,
                       params: ScenarioParams = ScenarioParams(),
                       grid: GridParams = GridParams()) -> list[Scenario]   # aynı seed → aynı banka
def default_generator(split: str = "train", data_path=DEFAULT_DATA_PATH,
                      params: ScenarioParams = ScenarioParams(),
                      grid: GridParams = GridParams()) -> ScenarioGenerator  # split ∈ {"train","test"}
```
`Scenario` tipi `config.py`'dedir. Geçersiz `split` → `ValueError`.

### 3.4 `envs/microgrid_lfc_env.py`
```python
class MicrogridLFCEnv(gym.Env):
    def __init__(self, scenario_generator: ScenarioGenerator | None = None,
                 grid: GridParams = GridParams(), bess: BessParams = BessParams(),
                 reward: RewardWeights = RewardWeights(), obs_scales: ObsScales = ObsScales()): ...
```
- `scenario_generator=None` → `default_generator("train")`.
- `reset(seed=None, options=None)`: `options={"scenario": Scenario}` verilirse o senaryo kullanılır,
  yoksa `scenario_generator.sample(self.np_random)`. Bölüm uzunluğu = `scenario.n_steps`.
  **Tüm rastgelelik `self.np_random` üzerinden**; global `np.random` kullanılmaz.
- `step(action)` → `(obs, reward, terminated, truncated, info)`; `terminated` **her zaman False**,
  `truncated` adım sayacı `n_steps`'e ulaşınca True. `reward` Python `float`; `terminated`, `truncated`
  ve `info["shield_active"]` Python `bool` (numpy tipi değil).
- `action_space = Box(-1, 1, (1,), float32)`; `observation_space = Box(-1, 1, (5,), float32)`.
- Açık öznitelikler (salt okunur kullanım): `grid`, `bess`, `dt`, `n_steps`, `step_count`,
  `delta_f_hz`, `rocof_hz_s`, `p_ess`, `u_ess`, `soc`, `scenario`.
- `info` — **hem `reset` hem `step`** şu anahtarları döndürür:
  `delta_f_hz`, `rocof_hz_s`, `p_ess` (gerçekleşen, pu), `u_ess` (uygulanan komut, pu), `soc`,
  `p_diesel` (ΔP_m, pu), `d_net` (o adımda uygulanan bozucu, pu; reset'te 0.0), `shield_active` (False).
- Bölüm sonunda (`truncated=True`) ek olarak: `ep_rmse_hz`, `ep_throughput_pu_s` (Σ|P_ess|·dt).

### 3.5 `agents/shield.py`
```python
class ShieldedEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, params: ShieldParams = ShieldParams()): ...
```
- Karar, adımdan **önce** ölçülen `env.unwrapped.delta_f_hz` ile verilir.
- Histerezis: pasifken `|Δf| > engage_hz` → aktif; aktifken `|Δf| < release_hz` **ve** en az `min_hold_s` aktif kalınmışsa → pasif.
- Aktifken ajan aksiyonu yerine PI çıkışı uygulanır:
  `a = clip(kp·e + ki·I, −1, 1)`, `e = −Δf_hz`, `I += e·dt` (doyumda integral dondurulur).
- Darbesiz geçiş: devreye girişte `I`, PI çıkışı o anki `u_ess / p_max`'a eşit olacak şekilde başlatılır.
- PI çıkışı ortamın aynı hız/güç/SOC sınırlarından geçer (wrapper yalnızca aksiyonu değiştirir).
- `info["shield_active"]` (bool) her adımda yazılır; aktifken ödülden `params.penalty` düşülür.
  Cezasız görev ödülü `info["reward_task"]` olarak da yazılır.
- Bölüm sonunda `info["ep_autonomy"]` ∈ [0, 1] (kalkanın pasif olduğu adım oranı).
- `reset()` kalkan durumunu sıfırlar.

### 3.6 `agents/baselines.py`
```python
class Controller(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...   # shape (1,), float32, [-1, 1]

class NoBessController            # her zaman 0
class DroopController(k)           # a = −k·Δf_hz
class PIController(kp, ki, dt)     # e = −Δf_hz; anti-windup'lı
class PIDController(kp, ki, kd, dt)  # türev ölçümden: −kd·rocof_hz_s
class RLController(model, name)    # model.predict(obs, deterministic=True)
```
Klasik kontrolcüler fiziksel ölçümü **`info`'dan** okur (`delta_f_hz`, `rocof_hz_s`); `obs`'tan ölçek çevirmez.

### 3.7 `utils/rollout.py`
```python
def run_episode(env: gym.Env, controller: Controller, scenario: Scenario, seed: int = 0) -> dict[str, np.ndarray]
```
`env.reset(seed=seed, options={"scenario": scenario})`, sonra bölüm bitene kadar adımlar.
Her adımda `controller.act(obs, info)` çağrılır (ilk adımda `reset`'in döndürdükleriyle); bölüm başında `controller.reset()`.
Dönen sözlük (hepsi uzunluk `n_steps`; `k`. eleman `k`. adımın **sonucu**): `t = (k+1)·dt` [s], `delta_f_hz`, `p_ess`, `u_ess`, `soc`, `p_diesel`,
`d_net`, `shield_active` (bool), `reward`. `reward` **görev ödülüdür**: `info["reward_task"]` varsa o
(kalkan cezası hariç), yoksa adımın ödülü. Böylece `mean_cost` tüm kontrolcülerde aynı büyüklüktür;
kalkan etkinliği ayrıca `autonomy_pct` ile raporlanır.

### 3.8 `utils/metrics.py`
```python
def compute_metrics(traj: dict[str, np.ndarray], scenario: Scenario | None = None,
                    grid: GridParams = GridParams(), bess: BessParams = BessParams(),
                    settle_band_hz: float = 0.02) -> dict[str, float]
def aggregate(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float]
    # {"mean", "std", "ci_low", "ci_high"} — ortalamanın bootstrap %95 aralığı; NaN'ler dışlanır;
    # std örneklem std'si (ddof=1; <2 değer varsa 0.0); hiç geçerli değer yoksa hepsi NaN; aynı seed → aynı sonuç
def paired_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float]
    # {"mean_diff", "median_diff", "p_value"} — Wilcoxon işaretli sıra (a − b); tüm farklar 0 ise p_value=1.0
```
Metrik anahtarları ve tanımları §7'de.

### 3.9 `utils/plotting.py`
Her fonksiyon bir dosya kaydeder ve yolunu döndürür; `matplotlib` arka ucu `Agg`. Ekrana çizim yok.
Tek giriş noktası (`evaluate.py` yalnızca bunu çağırır):
```python
def make_all_figures(example_trajs: dict[str, dict[str, np.ndarray]],   # kontrolcü adı → run_episode çıktısı (aynı örnek senaryo)
                     example_scenario: Scenario,
                     delta_f_all: dict[str, np.ndarray],                # kontrolcü adı → tüm test senaryolarının Δf'i (birleştirilmiş, Hz)
                     results: dict,                                     # SPEC §3.12 sözlüğü
                     out_dir: str | Path) -> list[Path]                 # 7 dosyanın yolları, Fig1…Fig7 sırasıyla
```
Her şekil için ayrı fonksiyonlar modül içinde serbestçe tanımlanır.
Fonksiyonlar eksik kontrolcülere (ör. SAC modeli yok) ve boş Pareto taramasına dayanıklı olmalıdır.
Üretilecek dosyalar (`figures/`): `Fig1_Zaman_Serisi.png`, `Fig2_Yuk_Adimi_Zoom.png`,
`Fig3_Frekans_Histogram.png`, `Fig4_Enerji_SOC.png`, `Fig5_Kalkan_Seridi.png`,
`Fig6_Pareto.png`, `Fig7_Ozet_Tablo.png`.

### 3.10 `train.py`
```python
def make_env(seed: int, shielded: bool = True, split: str = "train") -> gym.Env   # Monitor(ShieldedEnv(MicrogridLFCEnv))
def train(algo: str, seed: int, total_steps: int, out_dir: str | Path = "models",
          log_dir: str | Path | None = "logs", shielded: bool = True, **overrides) -> Path
```
- `algo ∈ {"td3", "sac"}`; `device="auto"`; model `out_dir/{algo}_seed{seed}.zip`
  (kalkansız eğitim: `{algo}_noshield_seed{seed}.zip`) olarak kaydedilir ve yol döndürülür.
- `overrides`: SB3 kurucu argümanları (ör. `learning_starts`, `batch_size`) — duman testleri için.
- CLI: `python train.py --algo td3 --seed 0 --steps 150000 [--no-shield] [--w-u 0.1]`.
- Hata yutulmaz; `KeyboardInterrupt`'ta `*_interrupted.zip` kaydedilir.

### 3.11 `tune_baselines.py`
```python
def tune(n_trials: int = 150, n_scenarios: int = 20, seed: int = 0,
         out_path: str | Path = "results/baseline_gains.json") -> dict
def pareto_sweep(n_scenarios: int = EvalParams.n_scenarios, seed: int = EvalParams.seed,
                 out_path: str | Path = "results/pareto_pi.json") -> list[dict]
```
- Senaryolar **eğitim** bölmesinden; amaç = bölüm ortalaması `−reward` (RL ile aynı maliyet).
- `baseline_gains.json`: `{"droop_k", "pi_kp", "pi_ki", "pid_kp", "pid_ki", "pid_kd"}`.
- `pareto_pi.json`: PI kazanç ızgarası için, varsayılan olarak `evaluate.py` ile **aynı test bankasında** `{kp, ki, rmse_mhz, bess_throughput_kwh}` listesi.

### 3.12 `evaluate.py`
```python
def run_evaluation(n_scenarios: int = 40, seed: int = 20240, model_dir: str | Path = "models",
                   out_dir: str | Path = "results", fig_dir: str | Path | None = "figures",
                   gains_path: str | Path | None = "results/baseline_gains.json") -> dict
```
- Test bölmesinden `make_scenario_bank`; **her kontrolcü birebir aynı senaryoları** çalıştırır.
- Klasik kontrolcüler kalkansız ham ortamda; `TD3+Shield` / `SAC+Shield` `ShieldedEnv`'de;
  `TD3` (ablasyon) kalkansız eğitilmiş modelle ham ortamda.
- RL modelleri `model_dir`'den desenle bulunur: `td3_seed*.zip`, `sac_seed*.zip`, `td3_noshield_seed*.zip`.
  Bulunamayan kontrolcü atlanır (uyarı basılır). Birden çok tohum varsa senaryo başına metrik tohumlar
  üzerinden ortalanır (eşleştirme korunur) ve `per_seed` özeti eklenir.
- `out_dir/results.json` yazılır ve aynı sözlük döndürülür. NaN değerler `None`/`null` olarak saklanır
  (döndürülen sözlükte de); sözlük `json.loads(json.dumps(x)) == x` koşulunu sağlar. **Dosyada zaman damgası, mutlak yol, süre yoktur**;
  aynı girdilerle iki çalıştırma bayt düzeyinde aynı dosyayı üretir (`sort_keys=True`, sabit tohumlar).
- `fig_dir=None` ise grafik üretilmez; verilirse §3.9'daki 7 dosyanın tümü üretilir.
- CLI: `python evaluate.py [--n-scenarios N] [--update-readme]`. `--update-readme`, README'deki
  `<!-- RESULTS:START -->` … `<!-- RESULTS:END -->` arasını `utils/report.py` çıktısıyla değiştirir.

### 3.13 `utils/report.py`
```python
def render_results_table(results: dict) -> str     # results.json sözlüğünden Markdown tablo; deterministik
def update_readme(results: dict, readme_path: str | Path = "README.md") -> None
```
Tablo: satırlar metrikler, sütunlar `results["controllers"]`'daki kontrolcüler (`ALL_CONTROLLERS` sırasıyla),
hücreler `ortalama ± std`; `None` değerler `—`. Satırlar ve ondalık basamak: `rmse_mhz` (1), `max_dev_mhz` (1),
`band_100_pct` (1), `settling_time_s` (1), `bess_throughput_kwh` (2), `control_jitter` (1), `autonomy_pct` (1),
`mean_cost` (3). README'deki sonuç sayıları **yalnızca** bu tablodan gelir.

`results.json` biçimi:
```json
{
  "meta": {"n_scenarios": 40, "seed": 20240, "grid": {...}, "bess": {...}, "shield": {...}, "reward": {...}},
  "controllers": {
    "PI": {
      "metrics": {"rmse_mhz": {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0}, "...": {}},
      "per_scenario": {"rmse_mhz": [0, 0], "...": []}
    },
    "TD3+Shield": {"metrics": {}, "per_scenario": {}, "per_seed": {"0": {"rmse_mhz": 0}}}
  },
  "comparisons": {"TD3+Shield_vs_PI": {"rmse_mhz": {"mean_diff": 0, "median_diff": 0, "p_value": 1}}},
  "pareto": {"PI_sweep": []}
}
```
`comparisons`: var olan her RL kontrolcüsü için `{RL}_vs_PI` ve `{RL}_vs_PID`.
`pareto.PI_sweep`: `results/pareto_pi.json` varsa içeriği, yoksa boş liste.

## 4. Gözlem

`obs` (float32, 5 boyut, her bileşen ölçeklenip `[-1, 1]`'e kırpılır):

| i | Büyüklük | Ölçek |
|---|---|---|
| 0 | `Δf_hz` | `ObsScales.f_hz` |
| 1 | `rocof_hz_s = (Δf_k − Δf_{k−1}) / dt` | `ObsScales.rocof_hz_s` |
| 2 | `I_k = clip(I_{k−1} + Δf_k·dt, ±int_hz_s)` [Hz·s], `I_0 = 0` | `ObsScales.int_hz_s` |
| 3 | `p_ess` | `BessParams.p_max` |
| 4 | `soc − soc_ref` | `0.5` |

Tüm bileşenler adım **sonrası** değerlerle hesaplanır.

## 5. Senaryo üretimi

Her bölüm için (yalnızca verilen `rng` ile):
1. Rüzgâr seviyesi `L`: `wind_levels`'tan rastgele bir saat (`meta["hour_index"]`).
2. Türbülans: OU süreci `x`, `x_0 = 0`, durağan std `turb_intensity`, zaman sabiti `turb_tau_s`;
   `ΔP_wind = wind_share · L · x`.
3. `ramp_prob` olasılıkla rüzgâr rampası: büyüklük `±U(ramp_size) · wind_share`, süre `U(ramp_duration_s)`,
   bölüm içinde rastgele başlangıç; rampadan sonra sabit kalır.
4. Yük: `n ~ U{n_load_steps}` adet adım; zamanlar `load_step_window_s` içinde, aralarında en az
   `load_step_min_gap_s`; büyüklük `±U(load_step_size)`. Üstüne `N(0, load_noise_std)` gürültü.
5. `soc0 ~ U(soc0_range)`.
6. `d_wind[0] = d_load[0] = 0` (bölüm dengede başlar).

## 6. Ödül

```
ef = Δf_hz / f_ref_hz
eu = p_ess / p_max
ej = (u_k − u_{k−1}) / p_max
es = (soc − soc_ref) / 0.5
r  = −(w_f·ef² + w_u·eu² + w_j·ej² + w_soc·es²)
```
Adım **sonrası** değerlerle hesaplanır. Dengede `r = 0`, her zaman `r ≤ 0`.
`w_soc` varsayılan 0'dır: AGC'nin boşaltma terimi yüzünden kalıcı batarya gücü frekans ofseti
(`Δf = P_ess/β`) yaratır; SOC bu modelde amaç değil **kısıt ve raporlanan metriktir**.
Kalkan aktifken `ShieldedEnv` ayrıca `penalty` düşer. Referans kontrolcü ayarının maliyeti = bölüm ortalaması `−r`.

## 7. Metrikler (`compute_metrics`)

| Anahtar | Tanım |
|---|---|
| `rmse_mhz` | `sqrt(mean(Δf²))·1000` |
| `mae_mhz` | `mean(|Δf|)·1000` |
| `max_dev_mhz` | `max(|Δf|)·1000` |
| `nadir_mhz`, `zenith_mhz` | `min(Δf)·1000`, `max(Δf)·1000` |
| `band_50_pct`, `band_100_pct`, `band_150_pct` | `|Δf| <` 50/100/150 mHz olan adım yüzdesi |
| `rocof_max_hz_s`, `rocof_mean_hz_s` | `|diff(Δf)|/dt` en büyük ve ortalama |
| `settling_time_s` | `scenario.meta["load_steps"]`'teki **ilk** adım için: adım anından, pencere içinde `|Δf| ≥ settle_band_hz` olan **son** örneğe kadar geçen süre + `dt`. Pencere: [adım, sonraki adım veya bölüm sonu). Bant hiç aşılmadıysa 0; pencere sonunda hâlâ dışarıdaysa NaN; senaryo/adım yoksa NaN |
| `bess_throughput_kwh` | `Σ|p_ess|·s_base_mw·1000·dt/3600` |
| `bess_efc` | `bess_throughput_kwh / (2·e_mwh·1000)` (eşdeğer tam çevrim) |
| `soc_min`, `soc_max`, `soc_final` | SOC yörüngesinden |
| `diesel_travel_pu` | `Σ|diff(p_diesel)|` (dizel regülasyon hareketi) |
| `control_jitter` | `Σ|diff(u_ess)| / p_max` |
| `autonomy_pct` | `100·(1 − mean(shield_active))` |
| `mean_cost` | `−mean(reward)` |

CO₂ metriği yoktur: batarya enerji geçişi emisyon değildir.
