# TASKS — İş paketleri

Sözleşme: `docs/SPEC.md` (arayüzler) + `config.py` (sabitler ve tipler) + `tests/acceptance/` (bitti tanımı).
Her paket **yalnızca kendi dosyalarına** yazar. Başka paketin dosyasında hata görürseniz düzeltmeyin, raporlayın.

Python: `C:\Users\aglamazlarefe\miniconda3\envs\rwfc\python.exe` (aşağıda `PY`).

## Aşama 1 — Çekirdek modüller (paralel)

| Paket | Dosyalar | Bitti ölçütü |
|---|---|---|
| **A — Dinamik + ortam** | `envs/dynamics.py`, `envs/microgrid_lfc_env.py` (yeniden yazılır), `tests/test_env_extra.py` (isteğe bağlı) | `PY -m pytest tests/acceptance/test_dynamics_and_env.py -q` |
| **B — Veri + senaryo** | `utils/data_loader.py` (`git mv utils/Data_loader.py`), `envs/scenarios.py` | `PY -m pytest tests/acceptance/test_data_and_scenarios.py -q` |
| **C — Kalkan + referanslar + rollout** | `agents/shield.py`, `agents/baselines.py`, `utils/rollout.py`; `agents/safe_hybrid_controller.py` silinir | `PY -m pytest tests/acceptance/test_shield_and_baselines.py -q` (A bittikten sonra geçer) |
| **D — Metrikler + grafikler** | `utils/metrics.py`, `utils/plotting.py` | `PY -m pytest tests/acceptance/test_metrics.py -q` |

Notlar:
- A, `envs/scenarios.py` hazır olmadan da çalışabilmeli: testlerin çoğu senaryoyu `options={"scenario": ...}` ile verir.
  `scenario_generator=None` durumunda `default_generator("train")` **tembel** (ilk ihtiyaçta) içe aktarılabilir.
- C, ortam hazır olmadan kendi testlerini çalıştıramaz; SPEC'e göre yazar, sahte (stub) ortamla birim testi yapabilir.
  Asıl doğrulama Aşama 2'dedir.
- D'nin grafik fonksiyonları `evaluate.py`'nin elindeki veriyle çağrılır: kontrolcü adı → yörünge sözlükleri
  (`run_episode` biçimi), `results` sözlüğü (SPEC §3.12) ve Pareto listesi. İmzaları modül docstring'ine yazın.
  Grafik metinleri Türkçe; arka uç `Agg`; açık tema, 200 dpi; renkler kontrolcü başına sabit.

## Aşama 2 — Bütünleştirme (tek ajan)

- `PY -m pytest tests/acceptance -q --deselect tests/acceptance/test_pipeline.py` tümüyle geçene kadar
  arayüz uyumsuzluklarını giderir (bu aşamada tüm Aşama 1 dosyalarına yazabilir).
- Model doğrulaması: `scripts/validate_model.py` yazar ve çalıştırır — yük adımı büyüklüklerine göre
  NoBESS / Droop / PI / PID / sıfır-ajan+kalkan için nadir, yerleşme, enerji geçişi, kalkan oranı tablosu.
  Çıktıyı `docs/MODEL_VALIDATION.md`'ye yazar.
- Parametre değişikliği gerekiyorsa (`config.py`) **yalnızca kabul testlerini bozmayacak** şekilde yapar ve
  gerekçesini `docs/MODEL_VALIDATION.md`'ye yazar. Kabul testlerini değiştirmez.
- Eski dosyaları temizler: `generate_figures.py`, `results.txt`, `agents/safe_hybrid_controller.py`, `utils/Data_loader.py`.

## Aşama 3 — Betikler (paralel)

| Paket | Dosyalar | Bitti ölçütü |
|---|---|---|
| **E — Eğitim** | `train.py` (yeniden yazılır) | `test_pipeline.py` içindeki `train` / `make_env` / `loads_and_predicts` testleri |
| **F — Referans ayarı** | `tune_baselines.py` | `test_pipeline.py::test_tune_and_pareto` |
| **G — Değerlendirme + rapor** | `evaluate.py` (yeniden yazılır), `utils/report.py` | `test_pipeline.py` içindeki `evaluation_*` ve `report_*` testleri (E bittikten sonra geçer) |

Notlar:
- E: TD3 ve SAC için CPU'ya uygun varsayılanlar (`net_arch=[256, 256]`, `train_freq=1`, `gradient_steps=1`,
  `learning_starts=5000`, TD3 için `NormalActionNoise(σ=0.1)`). `EvalCallback` **test** bölmesinde değil, eğitim
  bölmesinden sabit tohumlu ayrı bir doğrulama ortamında çalışır; en iyi model `out_dir/best/{ad}/` altına.
  `log_dir=None` iken TensorBoard ve EvalCallback kapalıdır (duman testleri hızlı kalsın).
  Her koşu `torch.set_num_threads(1)` ile başlar (paralel koşular çekirdekleri paylaşsın diye).
  Episode sonu metrikleri (`ep_rmse_hz`, `ep_autonomy`, `ep_throughput_pu_s`) TensorBoard'a yazılır.
- F: Optuna `TPESampler(seed=seed)`, sessiz günlük. Arama aralıkları kararlı bölgeyi kapsar (`docs/MODEL_VALIDATION.md`).
  Kararsız kazançlar zaten büyük maliyet üretir; ayrıca sınır koymaya gerek yok.
- G: SB3 modelleri `device="cpu"` ile yüklenir. RL kontrolcüleri için tohum = dosya adındaki sayı.

## Aşama 4 — Bağımsız inceleme (tek ajan, salt okunur)

Kodu SPEC'e karşı denetler: sihirli sayı, global `np.random` kullanımı, yutulmuş istisna, birim hatası,
test bölmesinin eğitime sızması, kalkanın iki yerde uygulanması. **Kod değiştirmez**; bulguları dosya:satır ile raporlar.

## Aşama 5 — Ana oturum

1. `PY tune_baselines.py` → `results/baseline_gains.json`, `results/pareto_pi.json`
2. Hız ölçümü, sonra arka planda eğitimler: TD3 ×3 tohum, SAC ×3 tohum, TD3 kalkansız ×1
3. `PY evaluate.py --update-readme`
4. README yeniden yazımı (model, parametreler, sonuç yorumu, sınırlılıklar)
5. `PY scripts/check_goal.py`
