# HANDOFF — Bir sonraki oturum için durum notu

Son güncelleme: 2026-11-27 ~10:30, Opus oturumundan Sonnet oturumuna devir.
Önce `CLAUDE.md`, sonra bu dosya. Arayüz sözleşmesi `docs/SPEC.md`, bitti tanımı `docs/GOAL.md`.

## 1. Neredeyiz (tek paragraf)

Fiziksel model yeniden yazımı **tamam ve test edildi**: `physical-model` dalı, 69/69 test geçiyor
(kabul testleri dahil), bağımsız inceleme bulguları düzeltildi. Kalan iş yalnızca **koşular + rapor**:
RL eğitimleri arka planda sürüyor, ardından değerlendirme, README sonuç yorumu ve `check_goal.py`.

## 2. Arka planda çalışan / çalışmış işler

| İş | Çıktı | Durum (10:30) |
|---|---|---|
| 7 RL eğitimi, 150 bin adım (TD3 ×3, SAC ×3, kalkansız TD3 ×1) | `models/{td3,sac}_seed{0,1,2}.zip`, `models/td3_noshield_seed0.zip`, günlük `logs/runs/*.log`, TensorBoard `logs/*_1/` | sürüyor; TD3 ~70k, SAC ~42k adımda. Kalan: TD3 ~25 dk, SAC ~55 dk |
| Kazanç ayarı (2. tur, `ki` aralığı 20'ye genişletildi) | `results/baseline_gains.json`, `results/pareto_pi.json`, günlük `logs/runs/tune2.log` | sürüyor (~20 dk) |

Durumu öğrenmek için:
```bash
ls models/*.zip                       # 7 dosya olunca eğitimler bitti
tasklist | grep -ci python            # çalışan python süreci sayısı
tail logs/runs/tune2.log              # ayar bittiyse kazanç tablosu burada
```
Eğitim ilerlemesi (TensorBoard'dan adım sayısı ve doğrulama ödülü):
```bash
PY -c "import glob;from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
for d in sorted(glob.glob('logs/*_1')):
  e=E(d,size_guidance={'scalars':0});e.Reload();t=e.Tags()['scalars']
  print(d,max(e.Scalars(x)[-1].step for x in t),[round(v.value,1) for v in e.Scalars('eval/mean_reward')] if 'eval/mean_reward' in t else '')"
```

**Süreçler çökerse / bilgisayar kapanırsa:** eğitimi yeniden başlatın (kaldığı yerden devam yok):
`PY train.py --algo sac --seed 0 --steps 150000` (her eksik model için). Paralel çalıştırırken **her komutu ayrı
satırda `&` ile başlatın**; `A && B && (C) &` biçimi tüm zinciri arka plana atar ve değişkenler kaybolur (bir kez başımıza geldi).

### Ek iş akışı (ultracode, 10:40'ta başlatıldı) — `wf_c2f2bea6-c84`

Yalnızca **yeni dosya** yazan sonnet ajanları; mevcut dosyalara dokunmazlar:

| Paket | Yeni dosyalar | Eğitimler bitince yapılacak |
|---|---|---|
| Öğrenme eğrileri | `scripts/plot_learning_curves.py` | `PY scripts/plot_learning_curves.py` → `figures/Fig8_Ogrenme_Egrileri.png`, `results/learning_curves.json`; README şekil tablosuna ekle |
| Dayanıklılık | `scripts/robustness.py`, `tests/test_robustness_smoke.py` | `PY scripts/robustness.py` → `results/robustness.json`, `figures/Fig9_Dayaniklilik.png`; README'ye kısa bölüm |
| Ek testler | `tests/test_extra_physics.py`, `tests/test_extra_components.py` | `xfail(strict=True, reason="BUG: …")` işaretli test varsa = gerçek hata; düzeltip işareti kaldır |
| Hakemler (fizik/kontrol, istatistik/protokol) + çürütme doğrulaması | dosya yok, yalnızca rapor | doğrulanmış (`verdict.refuted == false`) bulguları değerlendir/düzelt |

Sonuçlar: `C:\Users\aglamazlarefe\claude2\config\projects\D--RW-FC\507f25e3-4c64-418c-86b4-e9e6e0ade4b3\subagents\workflows\wf_c2f2bea6-c84\journal.jsonl`
(her ajan için bir `{"type":"result",...}` satırı). Yeni dosyaları gözden geçirip `PY -m pytest -q -m "not slow"` ile denedikten sonra commit'leyin.

## 3. Sıradaki adımlar (sırayla)

1. **Ayar sonucunu kontrol et** (`logs/runs/tune2.log`): hiçbir kazanç `tune_baselines.py::SearchSpace` sınırına
   dayanmamalı. 1. tur (`ki ≤ 5`) sonucu `logs/runs/baseline_gains_ki5.json`'da: PID `ki=4,998` sınıra dayanmıştı.
   Yine dayanırsa aralığı genişletip `PY tune_baselines.py` yeniden çalıştırın.
2. **Eğitimler bitince:** `bash scripts/finish_pipeline.sh` — modelleri bekler, `evaluate.py --update-readme`
   çalıştırır, `check_goal.py` ile denetler. (~5–10 dk)
3. **Şekillere bakın** (`figures/Fig1…Fig7`, Read aracıyla): okunaklı mı, etiketler çakışıyor mu?
4. **README'deki `<!-- YORUM -->` satırını** gerçek sonuç yorumuyla değiştirin (aşağıdaki §5 kurallarına göre).
   Tablo otomatik gelir; elle sayı yazmayın, yorumdaki sayılar tablodan/`results.json`'dan alınmalı.
5. `PY scripts/check_goal.py` → "HEDEF TAMAM". Sonra commit + push.
6. **PR'ı bitir:** taslak PR #1 (https://github.com/aglamazlarefe/renewable_frequency_controll/pull/1).
   PR açıklamasındaki test planı kutularını işaretle (`gh pr edit 1 --body-file …`), sonuç özetini ekle, sonra
   `gh pr ready 1` ve `gh pr merge 1 --merge` (squash değil; commit geçmişi korunur). `main`'e force-push yok.

**Git yetkisi (kullanıcı, 2026-11-27):** commit, push ve PR birleştirme **onay almadan** yapılabilir.
Birleştirme yalnızca doğrulanmış durumda: testler yeşil ve `check_goal.py` "HEDEF TAMAM".

## 4. Verilmiş kararlar (yeniden tartışmayın)

- **Tam fiziksel model** (kullanıcı seçimi): per-unit salınım + dizel governor/türbin + AGC, kesin ZOH.
- **Kalkan eşiği 200 mHz** (ilk 150 idi). Gerekçe `docs/MODEL_VALIDATION.md` §4: varsayılan PI 150 mHz'i
  senaryoların %32–50'sinde aşıyordu; 200 mHz ENTSO-E azami kalıcı sapması ve PI/PID'in en kötüsünün üstünde.
- **BESS hız sınırı 2 pu/s** (ilk 0,4): 0,4'te orta kazançlı PI bile ±600 mHz limit çevrime giriyordu.
- **`w_soc = 0`**: AGC bataryayı zaten boşaltıyor; SOC kısıt ve raporlanan metrik.
- **Metrikler cezasız görev ödülüyle**: `ShieldedEnv` `info["reward_task"]` yazar, `run_episode` onu kaydeder;
  `mean_cost` tüm kontrolcülerde aynı büyüklük. Eğitim ödülü cezayı içerir.
- **Doğrulama ortamı** sabit eğitim-bölmesi bankasını tekrarlar (`train.py::make_eval_env`), test bölmesi yalnız
  `evaluate.py` ve `pareto_sweep`'te.
- **Pareto taraması** değerlendirmeyle aynı bankada (40 senaryo, tohum 20240).
- Eski sürümün modelleri/günlükleri kaldırıldı; geri getirme `docs/LEGACY.md`.
- Kullanıcı tercihi: otonom çalış, Türkçe iletişim, GPU yok (her şey CPU); commit/push/PR birleştirme onaysız.

## 5. Sonuç yorumunu yazarken

- **Sonuç ne ise o.** Erken gözlem: 70k adımda TD3 doğrulama bölüm ödülü ≈ −53 (kalkan cezası dahil); ayarlı PI'ın
  eğitim bankası maliyeti 0,0116/adım ≈ −17/bölüm. RL klasik kontrolcüyü geçmiyorsa README bunu açıkça söylemeli.
- Karşılaştırmayı `results.json → comparisons` (Wilcoxon p değerleri) ve Pareto şekli üzerinden yapın:
  RL noktası PI eğrisinin altında/solunda mı (daha az enerjiyle daha düşük RMSE)?
- Özerklik (%), kalkanın ne sıklıkta devreye girdiğini gösterir; kalkansız TD3 ablasyonu kalkanın katkısını ölçer.
- CO₂ gibi savunulamayan iddia eklemeyin.
- RL zayıf çıkarsa, kullanıcıya **seçenek olarak** önerin (kendiniz başlatmayın, saatler sürer):
  daha uzun eğitim (500k adım), RL hiperparametre araması, gözleme bozucu tahmini eklemek.

## 6. Bilinen tuzaklar

- Sistem `python` (3.13) projeyi çalıştıramaz: hep `C:\Users\aglamazlarefe\miniconda3\envs\rwfc\python.exe`.
- Bilgisayar uykuya geçince koşular durur (03:08–10:23 arası durdu). Kullanıcıya önerildi:
  `powercfg /change standby-timeout-ac 0` (sistem ayarı — kullanıcı onayı olmadan değiştirmeyin).
- CPU 8 süreçle doluyken pytest çok yavaşlar; ağır testleri koşular bitince çalıştırın.
- `tests/acceptance/` ve `scripts/check_goal.py` değiştirilmez.
- `evaluate.py`, `results/baseline_gains.json` yoksa UYARI basar ve ayarsız varsayılanlarla çalışır — sonucu geçersiz yapar.

## 7. Commit geçmişi (physical-model dalı)

```
12fa3c6 Point CLAUDE.md to the legacy restore notes
f10aa7f Remove legacy TensorBoard logs and models incompatible with the new env
5a08251 Rewrite on a physical microgrid model with shield, baselines and evaluation
0e2a13e Remove superseded scripts and rename data loader ahead of the rewrite
20139c6 Add contract for physical-model rewrite: config, spec, acceptance tests
```
Commit edilmemiş (bu devirle birlikte commit'lendi): README yeniden yazımı, `tune_baselines.py` ki aralığı,
bu dosya, `scripts/finish_pipeline.sh`.
