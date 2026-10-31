# Frozen protocol records

These declarations and launchers preserve the exact clean-v6 and official-v6 experiment sequence and fail-closed checks.

The `run.sh` files are archival protocol records, not publication-layout entry points: their directory-relative paths use the original experiment layout under `5/experiments/...`. Use the commands in the root [`README.md`](../../README.md) and the organized entry points under `scripts/`.

All required checkpoints and probes are available from the publication's `v1-paper` release. The pinned TRAIN and official caches are not included and must be regenerated deterministically before the corresponding protocol can run.
