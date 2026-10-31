# V1 exact-K4 matched pooling retention

Frozen V1 seed/checkpoint: `424242` / `989f9bc496bea484974679cd181e43836caf3283ffdc8e66507046945e0902de`.
All three separately fitted probes use only the parent, the same architecture, tensor-identical initialization, and the same 400-update TRAIN-only minibatch schedule. The evaluation population is the published official-v6 safe exact-K4 subset.

| Method | Mean ordered-child cosine |
|---|---:|
| V1 learned parent | 0.908104758 |
| Token-mass-weighted normalized mean | 0.910180213 |
| Unweighted normalized mean | 0.912319726 |

## Per-corpus means

| Corpus | Rows | V1 | Weighted | Unweighted |
|---|---:|---:|---:|---:|
| fineweb_edu | 148 | 0.903224968 | 0.905819906 | 0.908368401 |
| fineweb | 133 | 0.902914000 | 0.904951341 | 0.906805336 |
| openwebtext | 142 | 0.909281185 | 0.911209136 | 0.913876899 |
| wikipedia_en_20231101 | 183 | 0.914910917 | 0.916708400 | 0.918314761 |

| Paired difference | Mean | 95% document bootstrap CI |
|---|---:|---:|
| V1 - weighted | -0.002075455 | [-0.002226628, -0.001935234] |
| V1 - unweighted | -0.004214968 | [-0.004710167, -0.003748746] |

Weighted comparison: Pooling is higher under this matched protocol.
Unweighted comparison: Pooling is higher under this matched protocol.

Under this specified nonlinear matched parent-only probe protocol, both pooling baselines preserve more probe-recoverable ordered child information than V1. The required legacy probe is residual around its parent input (and initializes to the parent), so this result is specific to that architecture and may favor mean-like representations. The experiment does not establish lossless encoding, preserved mutual information, architecture-independent recoverability, universal downstream superiority, or semantic superiority.

Official exact-K4 documents: **606**. Audit passed: **True**.

## Reproduction

```bash
# Preserve the committed machine-readable evidence before a create-only reproduction:
mv results/v1_k4_matched_pooling results/v1_k4_matched_pooling.published
# Materialize the release checkpoint:
# Obtain the exact final checkpoint through the anonymous submission discussion.
python supplement.py materialize /path/to/weights --role final_clean_v6_checkpoint --mode hardlink
PYTHONPATH=src python scripts/final_v6/prepare_train.py --config configs/final_v6/seed424242.yaml --device cuda
PYTHONPATH=src python scripts/v1_k4_pooling.py prepare-official --device cuda
PYTHONPATH=src python scripts/v1_k4_pooling.py fit --device cuda
PYTHONPATH=src python scripts/v1_k4_pooling.py evaluate --device cuda
PYTHONPATH=src python scripts/v1_k4_pooling.py audit
PYTHONPATH=src python scripts/v1_k4_pooling.py report
```

If the original `cache_kary4_strict_autoencoder_official_test_v6` is present, omit `prepare-official`; evaluation reuses it preferentially.
