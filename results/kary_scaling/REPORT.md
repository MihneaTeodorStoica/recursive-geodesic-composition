# K=8 with 768-token leaves — validation only

No held-out official benchmark artifact was accessed. All arms were evaluated on the exact same new K8 validation cache rows.

## Configuration

- True contiguous arity: **8**; fixed leaves: **768 tokens**; random leaves: **384..768 tokens**.
- First-level teacher span: **8 × 768 = 6,144 tokens exactly**. Longer targets are omitted.
- Encoder: width 1,024, 8 layers, 16 heads, 768-D output; tangent-expmap angle 0.5 with semantic weighting.
- 1,200 updates; batch 8; equal-corpus stratification; 160-step warmup and cosine LR; validation batch 32.

## Validation

| Arm | Arity | pooled logistic AUC | pooled RBF AUC | orientation-safe strongest AUC | teacher cosine | mean/max depth | docs/s | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kary4_reference | 4 | 0.573077 | 0.592991 | 0.592991 | 0.978854 | 1.140/2 | 5256.61 | 111,538,433 |
| kary8 | 8 | 0.668336 | 0.660172 | 0.668336 | 0.977910 | 1.024/2 | 5255.50 | 124,133,633 |
| binary_reference | 2 | 0.631497 | 0.647113 | 0.647113 | 0.977828 | 1.597/4 | 4811.13 | 73,149,056 |

Selected K8 checkpoint training macro strongest AUC: raw **0.676218**, MA(3) **0.712988** at step **500**.
Pooled candidate logistic/RBF orientation-safe strongest AUC: **0.668336 / 0.660172 / 0.668336**.
Selected-checkpoint validation angle saturation fraction: **0.000000000**.
Depth reduction versus K4: **10.19%**; exact-row throughput ratio: **0.9998×**.
The saturation guard permanently rejected training at step **581** (batch saturation fraction **0.125000**). Training still completed all **1200** configured steps with checkpoint selection disabled; the preserved selected checkpoint is step **500**.

## Per-corpus K8 candidate

| Corpus | cosine | logistic AUC | RBF AUC | orientation-safe strongest AUC | segmentation cosine |
|---|---:|---:|---:|---:|---:|
| fineweb_edu | 0.978566 | 0.649427 | 0.616204 | 0.649427 | 0.986402 |
| fineweb | 0.975957 | 0.631742 | 0.604335 | 0.631742 | 0.983694 |
| openwebtext | 0.979252 | 0.699446 | 0.646010 | 0.699446 | 0.985973 |
| wikipedia_en_20231101 | 0.977865 | 0.724062 | 0.641348 | 0.724062 | 0.985719 |

## Matched parent-only retention

The same exact-8-child train/validation rows and identical ordered parent-only probe protocol were used for every arm.

| Arm | ordered child cosine | train rows | validation rows |
|---|---:|---:|---:|
| kary4_reference | 0.880711 | 388 | 77 |
| kary8 | 0.883688 | 388 | 77 |
| binary_reference | 0.878806 | 388 | 77 |

## K16 escalation gate

- FAIL — `strongest_auc_noninferior`
- PASS — `cosine_noninferior_and_floor`
- FAIL — `no_angle_saturation`
- FAIL — `depth_reduction_at_least_20pct`
- FAIL — `validation_root_speedup_at_least_15pct`
- PASS — `segmentation_noninferior`
- PASS — `retention_noninferior`

**K16 recommendation: NO**. K16 is recommended only if every AUC, cosine, saturation, depth, speed, segmentation, and matched-retention gate passes.
