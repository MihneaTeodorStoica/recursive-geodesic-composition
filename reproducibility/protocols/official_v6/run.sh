#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="$ROOT/experiments/kary4_strict_autoencoder_official_test_v6"
CONFIG="$STAGE/config.yaml"
CACHE="$ROOT/cache_kary4_strict_autoencoder_official_test_v6"
OUTPUT="$ROOT/results_kary4_strict_autoencoder_official_test_v6"
DEVICE="${DEVICE:-cuda}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$STAGE:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

PREACCESS_TAG="official-k4-strict-ae-v6-preaccess"
TAG_COMMIT="$(git rev-list -n 1 "$PREACCESS_TAG" 2>/dev/null || true)"
HEAD_COMMIT="$(git rev-parse HEAD)"
if [[ -z "$TAG_COMMIT" || "$TAG_COMMIT" != "$HEAD_COMMIT" ]]; then
  echo "Refusing launch: HEAD is not the annotated v6 preaccess tag." >&2
  exit 74
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing launch: tracked source differs from the frozen commit." >&2
  exit 75
fi

if [[ -e "$CACHE.PREPARATION_STARTED" || -e "$CACHE" || \
      -e "$OUTPUT.EVALUATION_STARTED" || -e "$OUTPUT" ]] || \
   compgen -G "$CACHE.staging.*" >/dev/null || \
   compgen -G "$OUTPUT.staging.*" >/dev/null; then
  echo "Refusing official-test-v6 rerun: marker/cache/result/staging already exists." >&2
  exit 73
fi

# Freeze-check protocol code/tests and every immutable checkpoint/provenance/probe
# input before any module is allowed to import datasets or inspect a v6 cache.
sha256sum --check "$STAGE/PREDECLARATION_SHA256SUMS"
/venv/main/bin/python "$STAGE/verify_preaccess.py"

# Each command claims its own exclusive marker and publishes only by atomic rename.
/venv/main/bin/python "$STAGE/prepare_official_test_v6.py" --config "$CONFIG" --device "$DEVICE"
/venv/main/bin/python "$STAGE/evaluate_official_test_v6.py" --config "$CONFIG" --device "$DEVICE"
