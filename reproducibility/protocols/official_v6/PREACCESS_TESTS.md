# Preaccess test record

All checks below were run before the annotated freeze tag. They were corpus/cache-free; `run.sh`, preparation, and evaluation were not launched.

## Focused protocol

- Command: `/venv/main/bin/pytest -q experiments/kary4_strict_autoencoder_official_test_v6/tests/test_official_test_protocol_v6.py`
- Outcome: **14 passed** in 17.42s.

The tests cover the exact scan/refusal partition, lazy corpus import, deterministic eligibility, exact tail coverage, direct/full validators, unrestricted deep recursion, every-internal-merge decoder telemetry, grouped fixed attacks, exact 2000-replicate source-group bootstrap, shared three-seed units, all gate components, atomic claims/storage failure, and the sole-target text invariant.

## Static checks

- `/venv/main/bin/ruff format --check experiments/kary4_strict_autoencoder_official_test_v6`: **5 files already formatted**.
- `/venv/main/bin/ruff check experiments/kary4_strict_autoencoder_official_test_v6`: **passed**.
- `python3 -m compileall -q experiments/kary4_strict_autoencoder_official_test_v6`: **passed**.
- `bash -n experiments/kary4_strict_autoencoder_official_test_v6/run.sh`: **passed**.

## Immutable preaccess verifier

`verify_preaccess.py` completed with `preaccess_verified=true` and `official_data_or_cache_opened=false`. It verified all three exact candidate and source checkpoint hashes, actual strict decoder state (19 decoder tensors per seed), finite model/probe tensors, genuine nonzero encoder/decoder changes, clean-development PASS/no-protected-access records, all six fresh-v6-TRAIN probe hashes, all frozen TRAIN-only floors, and the fixed attack hash.

## Full suite

An initial unconstrained recursive discovery command (`pytest -q`) stopped during collection because old experiment directories reuse the top-level module name `test_protocol`. An importlib-mode attempt exposed the corresponding old bare-import collisions. Neither failure executed a test or changed an artifact.

The canonical explicit project suite was then run with the committed clean-v6 development protocol test and the new official-v6 test added:

`/venv/main/bin/pytest -q test_*.py experiments/kary4_strict_autoencoder_v6_clean_development/test_protocol.py experiments/kary4_strict_autoencoder_official_test_v6/tests/test_official_test_protocol_v6.py --deselect=test_validation_output_calibration.py::test_checkpoint_and_transform_hashes_are_verifiable`

Outcome: **214 passed, 1 deselected, 11 warnings** in 24.23s. The single deselection is the already documented legacy storage test whose old encoder checkpoint was deliberately removed before clean-v6; the current exact v6 checkpoints and probes were separately hash- and tensor-verified above.

## Freeze-state checks

- New official cache, preparation marker, result, and evaluation marker: **all absent**.
- Paper diff: **empty**.
- Existing unrelated tracked/untracked artifacts: **not modified or staged**.
