#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="$ROOT/experiments/kary4_strict_autoencoder_official_test_v6"
OUTPUT="$ROOT/results_kary4_strict_autoencoder_official_test_v6"
DEVICE="${DEVICE:-cuda}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$STAGE:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

[[ -f "$OUTPUT.EVALUATION_STARTED" ]] || { echo "Missing spent initial marker" >&2; exit 73; }
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.EVALUATION_RECOVERY_STARTED" ]] || { echo "Recovery/output already exists" >&2; exit 73; }
if compgen -G "$OUTPUT.staging.*" >/dev/null; then
  echo "Refusing recovery with result staging residue" >&2
  exit 73
fi
[[ -f "$ROOT/cache_kary4_strict_autoencoder_official_test_v6/COMPLETE" ]] || { echo "Missing complete prepared cache" >&2; exit 73; }
sha256sum --check "$STAGE/RECOVERY_SHA256SUMS"
/venv/main/bin/python "$STAGE/evaluate_official_test_v6_recovery.py" --config "$STAGE/config.yaml" --device "$DEVICE"
