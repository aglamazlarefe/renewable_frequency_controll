# Eski sürümden kaldırılan dosyalar

Fiziksel model yeniden yazımında (`physical-model` dalı), eski sürüme ait aşağıdaki dosyalar depodan kaldırıldı.
Yeni gözlem uzayıyla (5 boyut) ve yeni ortamla uyumsuzlar. Dosyalar **silinmedi**, git geçmişinde duruyorlar.

## Kaldırılan dosyalar

| Dosya | İçerik |
|---|---|
| `logs/TD3_1/` … `logs/TD3_11/` | Eski TD3 eğitimlerinin TensorBoard günlükleri |
| `logs/evaluations.npz` | Eski EvalCallback çıktısı |
| `models/td3_microgrid_model.zip` | Eski ana model (3 boyutlu gözlem) |
| `models/td3_microgrid_model_interrupted.zip` | Eski yarıda kesilmiş model |
| `models/checkpoints/td3_mg_100000_steps.zip` | Eski 100 bin adım checkpoint'i |

Bu dosyalar şu iki commit'te hâlâ bulunuyor:
- `9524ce6`: `main` dalının ucu, eski sürümün son hâli
- `5a08251`: kaldırmadan hemen önceki `physical-model` commit'i

Önceki yeniden yazım commit'leri (`0e2a13e`) ayrıca şunları kaldırdı: `generate_figures.py`, `results.txt`,
`agents/safe_hybrid_controller.py`. `utils/Data_loader.py` ise `utils/data_loader.py` olarak yeniden adlandırıldı.
Bunlar da `9524ce6`'dan aynı yöntemle geri getirilebilir.

## Geri getirme

Belirli dosyaları çalışma dizinine geri getirmek için (dal değişmez):

```bash
git checkout 9524ce6 -- models/td3_microgrid_model.zip
git checkout 9524ce6 -- logs/                      # tüm eski günlükler
git checkout 9524ce6 -- models/ logs/              # hepsi
```

Eski sürümün tamamını ayrı bir klasörde çalıştırmak için (mevcut çalışma dizinine dokunmaz):

```bash
git worktree add ../rwfc-eski main
cd ../rwfc-eski
```

Eski kodun yeni ortamla çalışmadığını unutmayın: eski modeller yalnızca eski `envs/microgrid_lfc_env.py` ile
yüklenip çalıştırılabilir. Eski kodun gerçek çıktısı `results/before_refactor.txt` dosyasında saklıdır.

Not: `figures/` altındaki eski şekiller (`Fig1_Ana_Karsilastirma.jpg` vb.) bu adımda kaldırılmadı; yeni
şekiller farklı adlarla üretilir.
