#!/usr/bin/env bash
# scripts/finish_pipeline.sh — Eğitimler bittikten sonra kalan adımları çalıştırır.
#
#   bash scripts/finish_pipeline.sh          # eğitimlerin bitmesini bekler, sonra değerlendirir
#
# Adımlar: (1) 7 model dosyası oluşana kadar bekle  (2) evaluate.py --update-readme
#          (3) scripts/check_goal.py (tam denetim)
# Arka planda çalıştırılabilir; çıktı logs/runs/finish.log'a da yazılır.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/c/Users/aglamazlarefe/miniconda3/envs/rwfc/python.exe
export PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
mkdir -p logs/runs
exec > >(tee -a logs/runs/finish.log) 2>&1

MODELS="td3_seed0 td3_seed1 td3_seed2 sac_seed0 sac_seed1 sac_seed2 td3_noshield_seed0"
echo "[$(date +%H:%M)] model dosyaları bekleniyor: $MODELS"
while true; do
  missing=""
  for m in $MODELS; do [ -f "models/$m.zip" ] || missing="$missing $m"; done
  [ -z "$missing" ] && break
  sleep 60
done
echo "[$(date +%H:%M)] tüm modeller hazır"

echo "[$(date +%H:%M)] kazanç ayarı (logs/runs/tune2.log) bekleniyor"
until grep -q "ayarlı" logs/runs/tune2.log 2>/dev/null; do sleep 30; done

if [ ! -f results/baseline_gains.json ]; then
  echo "HATA: results/baseline_gains.json yok — önce: $PY tune_baselines.py"; exit 1
fi
echo "--- ayarlı kazançlar:"; cat results/baseline_gains.json; echo

echo "[$(date +%H:%M)] değerlendirme başlıyor"
"$PY" evaluate.py --update-readme || { echo "HATA: evaluate.py başarısız"; exit 1; }

echo "[$(date +%H:%M)] hedef denetimi"
"$PY" scripts/check_goal.py
echo "[$(date +%H:%M)] bitti (çıkış kodu $?)"
