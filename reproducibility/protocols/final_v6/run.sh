#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /venv/main/bin/activate
EXP=experiments/kary4_strict_autoencoder_v6_clean_development
AGG=results_kary4_strict_autoencoder_v6_clean_development
TRAIN_CACHE=cache_kary4_strict_autoencoder_v6_clean_train
DEV_CACHE=cache_kary4_strict_autoencoder_v6_clean_development
RUN_MARKER="${AGG}.RUN_STARTED"
for path in "$AGG" "$TRAIN_CACHE" "$DEV_CACHE" "$RUN_MARKER"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 1; }
done
printf 'development_only=true\nprotected_data_accessed=false\n' > "$RUN_MARKER"
pytest -q test_*.py "$EXP/test_protocol.py" 2>&1 | tee "${AGG}.FULL_SUITE.log"
{
  ruff check --select F,E9 train.py "$EXP"
  python -m py_compile train.py "$EXP"/*.py
  git diff --check
} 2>&1 | tee "${AGG}.STATIC_CHECKS.log"
python "$EXP/prepare_train.py" --device cuda 2>&1 | tee "${AGG}.PREPARE_TRAIN.log"
for seed in 424242 271828 314159; do
  OUT="results_kary4_strict_autoencoder_v6_clean_development_seed${seed}"
  mkdir "$OUT"
  printf 'development_only=true\nfixed_final_step=400\nseed=%s\n' "$seed" > "$OUT/TRAINING_STARTED"
  python train.py --config "$EXP/configs_train_availability_recovery/seed${seed}.yaml" --variant autoencoder --seed "$seed" --device cuda \
    2>&1 | tee "$OUT/TRAIN_STDOUT.log"
done
python "$EXP/freeze_training_and_probes.py" --device cuda 2>&1 | tee "${AGG}.FREEZE.log"
mv "${AGG}.FULL_SUITE.log" "${AGG}.STATIC_CHECKS.log" "${AGG}.PREPARE_TRAIN.log" "${AGG}.FREEZE.log" "$AGG/"
python "$EXP/prepare_development.py" --device cuda 2>&1 | tee "$AGG/PREPARE_DEVELOPMENT.log"
for seed in 424242 271828 314159; do
  python "$EXP/evaluate_development.py" --config "$EXP/configs_train_availability_recovery/seed${seed}.yaml" --device cuda \
    2>&1 | tee "results_kary4_strict_autoencoder_v6_clean_development_seed${seed}/EVALUATE_STDOUT.log"
done
python "$EXP/finalize.py" 2>&1 | tee "$AGG/FINALIZE_STDOUT.log"
