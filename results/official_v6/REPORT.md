# Strict-K4 AUTOENCODER official test v6

This atomic one-shot evaluates the three confirmed decoder-bearing AE checkpoints on one untouched test population. The same first 1000 safe-prefix-length-eligible source documents per corpus feed both mandatory arms. No post-access tuning, selection, rerun, suppression, or updates occur.

The sole AUC target is the frozen strict bound `<0.60`; the per-seed guard is derived from that target and its declared margin.

## Safe teacher-comparable arm

| seed | strongest pooled attack | safe AUC | 95% source-group CI | pooled cosine | macro cosine | built-in exact4 pooled cosine | TRAIN probe/floor |
|---:|---|---:|---:|---:|---:|---:|---:|
| 424242 | standard_scaled_logistic | 0.582809 | [0.576087, 0.589997] | 0.985049 | 0.985049 | 0.915040 | 0.908098/0.897565 |
| 271828 | standard_scaled_logistic | 0.582124 | [0.575781, 0.588464] | 0.985033 | 0.985033 | 0.915027 | 0.908054/0.897558 |
| 314159 | standard_scaled_logistic | 0.587763 | [0.581316, 0.594269] | 0.985052 | 0.985052 | 0.915116 | 0.908151/0.897599 |

Three-seed mean strongest pooled point: **0.584232**; shared-heldout-group 2000-bootstrap 95% CI `[0.578232, 0.590338]`.

Pooled and per-corpus three-seed shared-group aggregate points and CIs are retained in `results.json`.

## Unrestricted full-document recursion arm

Every available token is represented by an exact contiguous <=1024-token leaf; there is no token, leaf, or depth cap and no truncation. This arm has no full-root teacher target, AUC, or teacher cosine.

| seed | tokens | leaves | max tokens/doc | max leaves/doc | max depth | merges | decoder active-child cosine | full root rank | docs/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 424242 | 9142043 | 11356 | 58731 | 58 | 3 | 4789 | 0.909709 | 768 | 130.927 |
| 271828 | 9142043 | 11356 | 58731 | 58 | 3 | 4789 | 0.909842 | 768 | 127.795 |
| 314159 | 9142043 | 11356 | 58731 | 58 | 3 | 4789 | 0.909591 | 768 | 131.447 |

Full per-corpus/depth/leaf-vs-recursive-child decoder cosine, every-depth geodesic/importance saturation, token/leaf/depth distributions, maxima, health, and throughput are retained in `results.json`.

## Reporting-only gate

- `three_seed_mean_strongest_pooled_point_below_sole_auc_target`: **True**
- `every_seed_strongest_pooled_point_below_derived_guard`: **True**
- `every_seed_pooled_teacher_cosine_strictly_above_required_value`: **True**
- `every_seed_every_safe_scope_builtin_exact4_child_cosine_at_least_required_value`: **True**
- `every_seed_every_frozen_train_probe_floor_pass`: **True**
- `no_safe_or_full_collapse_nonfinite_or_saturation`: **True**
- `full_arm_mandatory_and_complete`: **True**
- conjunction: **True** (reporting only).

## Provenance

Checkpoint and source hashes, actual `variant=autoencoder` and strict-decoder state proof, genuine nonzero encoder/decoder change, seed-specific fresh-v6-TRAIN probe identities and frozen TRAIN-only floors, exact source counts and hashes, test role, every cache-shard hash, environment, and timing are in `results.json`.
