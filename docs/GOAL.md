# GOAL — Tamamlanma koşulu

## Hedef (tek cümle)

Mikro şebeke frekans kontrol projesini, fiziksel olarak tutarlı bir model üzerinde, iyi ayarlanmış klasik
kontrolcülerle **adil ve tekrarlanabilir** biçimde karşılaştırılan TD3 ve SAC ajanlarıyla yeniden kurmak;
README'deki her sayı koddan üretilebilir olsun.

## `/goal` için kullanılabilir metin

```
/goal `C:\Users\aglamazlarefe\miniconda3\envs\rwfc\python.exe scripts/check_goal.py` komutu "HEDEF TAMAM"
yazdırıp 0 ile çıkana kadar çalış. tests/acceptance/ altındaki dosyaları ve scripts/check_goal.py'yi değiştirme;
bir test yanlışsa gerekçesini yaz ve dur. docs/SPEC.md sözleşmedir, docs/TASKS.md iş listesidir.
```

## Makinece doğrulanan koşullar (`scripts/check_goal.py`)

| # | Koşul |
|---|---|
| 1 | `pytest -q` sıfır hata (kabul testleri dahil) |
| 2 | `results/results.json` var ve 7 kontrolcünün tümünü içeriyor: NoBESS, Droop, PI, PID, TD3+Shield, SAC+Shield, TD3 |
| 3 | En az 30 test senaryosu; TD3+Shield ve SAC+Shield için en az 3'er eğitim tohumu |
| 4 | `results/baseline_gains.json` var (referanslar ayarlanmış) ve PI Pareto taraması dolu |
| 5 | `figures/` altında 7 grafik (`Fig1_…` – `Fig7_…`) |
| 6 | README'deki `<!-- RESULTS:START -->…<!-- RESULTS:END -->` bloğu `results.json`'dan üretilen tabloyla birebir aynı |
| 7 | `evaluate.py` aynı girdilerle iki çalıştırmada bayt düzeyinde aynı `results.json`'ı üretiyor |

## Makinece doğrulanamayan, elle bakılacak koşullar

- README'de model denklemleri, parametre tablosu, yeniden üretim komutları ve **sınırlılıklar** bölümü var.
- README sonucu olduğu gibi anlatıyor: RL klasik kontrolcüyü geçmiyorsa bunu açıkça söylüyor.
- CO₂ tasarrufu gibi savunulamayan iddia yok.
- Kodda sihirli sayı yok; tüm sabitler `config.py`'den geliyor.

## Kapsam dışı

- GPU eğitimi (her şey CPU'da çalışır).
- RL hiperparametreleri için tam Optuna araması.
- Uzak depoya `git push` (kullanıcı isteyene kadar yapılmaz).
