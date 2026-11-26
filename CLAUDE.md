# CLAUDE.md

Rüzgâr ağırlıklı izole mikro şebekede BESS ile ikincil frekans kontrolü: TD3/SAC ajanı + PI güvenlik kalkanı,
iyi ayarlanmış klasik kontrolcülerle (Droop/PI/PID) karşılaştırma. Hedef akademik düzeyde savunulabilir sonuç.

## Önce bunları okuyun

- `docs/SPEC.md` — model denklemleri ve **modül arayüzleri (sözleşme)**
- `docs/TASKS.md` — iş paketleri ve dosya sahipliği
- `docs/GOAL.md` — tamamlanma koşulu
- `config.py` — tüm sabitler ve paylaşılan tipler

## Ortam ve komutlar

GPU yok; her şey CPU'da çalışır. Sistem `python`'u (3.13) projeyi çalıştıramaz — her zaman `rwfc` ortamını kullanın:

```
PY = C:\Users\aglamazlarefe\miniconda3\envs\rwfc\python.exe

PY -m pytest -q                              # tüm testler
PY -m pytest -q -m "not slow"                # hızlı testler (eğitim/değerlendirme duman testleri hariç)
PY -m pytest tests/acceptance/test_metrics.py -q
PY tune_baselines.py                         # referans kazançları + PI Pareto taraması
PY train.py --algo td3 --seed 0 --steps 150000
PY evaluate.py --update-readme               # results/results.json + figures/ + README tablosu
PY scripts/check_goal.py                     # hedef doğrulayıcı
```

Komutlar proje kökünden çalıştırılır. Bash'te Türkçe çıktı için `PYTHONIOENCODING=utf-8`, grafik için `MPLBACKEND=Agg`.

## Kurallar

- **`tests/acceptance/` ve `scripts/check_goal.py` değiştirilmez.** Bir kabul testi yanlış görünüyorsa
  zayıflatmayın; gerekçesiyle raporlayın. Ek testler `tests/` altına serbestçe yazılabilir.
- **Sihirli sayı yok.** Fiziksel parametre, eşik ve ölçek sabitleri yalnızca `config.py`'de durur.
- **Global `np.random` yok.** Ortam `self.np_random`, diğer her şey açıkça verilen `np.random.Generator` kullanır.
- **Birimler:** güç pu (sistem tabanı), frekans dış arayüzde Hz, zaman s. `P_ess > 0` deşarj.
- **Hata yutulmaz.** `except Exception: pass` ya da hata durumunda sessizce sıfır döndürmek yok.
- **Kalkan tek yerde:** `agents/shield.py::ShieldedEnv`. Ortamın içinde ya da değerlendirme betiğinde ikinci bir kopyası olmaz.
- **Test verisi sızdırılmaz:** eğitim, doğrulama ve kazanç ayarı yalnızca `default_generator("train")` ile;
  test bölmesi sadece `evaluate.py` ve `pareto_sweep` içinde.
- **Tekrarlanabilirlik:** `results/results.json` içinde zaman damgası, süre, mutlak yol bulunmaz.
- README'deki sonuç sayıları elle yazılmaz; `python evaluate.py --update-readme` üretir.
- Yorumlar ve grafik metinleri Türkçe; tanımlayıcılar İngilizce.
- Eski `models/td3_microgrid_model*.zip` ve `models/checkpoints/` yeni gözlem uzayıyla uyumsuzdur; dokunmayın, kullanmayın.

## Git

Çalışma dalı `physical-model`. Yerel commit serbest; **`git push` yapmayın** (kullanıcı isteyene kadar).
`results/before_refactor.txt`: yeniden yazımdan önceki kodun gerçek çıktısı (README'deki eski rakamlarla uyuşmaz) — referans olarak saklanır.
