# V3 CLS downstream run report

Status: **complete**.

## Outputs

- Machine-readable metrics and full validation grids: `results/v3_cls/downstream/DOWNSTREAM_RESULTS.json`
- Immutable shared leaf cache and provenance: `results/v3_cls/downstream/leaf_cache/CACHE_MANIFEST.json`
- Cache completion hash: `results/v3_cls/downstream/leaf_cache/COMPLETE`
- Result completion hash: `results/v3_cls/downstream/DOWNSTREAM_COMPLETE`
- Frozen executable protocol: `scripts/v3_cls/downstream/protocol.json`
- Implementation: `scripts/v3_cls/downstream/prepare_leaf_cache.py`, `scripts/v3_cls/downstream/evaluate.py`
- Logs: `reproducibility/v3_cls/DOWNSTREAM_CACHE_PREPARATION.log`, `reproducibility/v3_cls/DOWNSTREAM_EVALUATION.log`

## ECtHR-A test results

All values below use the selected validation C (`C=100` for every method), refit on canonical train+validation, and evaluated once on canonical test. The primary metrics include ten violation labels and the derived no-violation label.

| representation | micro-F1 | macro-F1 | violation-only micro-F1 | violation-only macro-F1 |
|---|---:|---:|---:|---:|
| recursive 424242 (candidate) | 0.6347 | 0.5081 | 0.6683 | 0.5278 |
| recursive 271828 (final) | 0.6335 | 0.5083 | 0.6670 | 0.5302 |
| recursive 314159 (final) | 0.6301 | 0.5342 | 0.6650 | 0.5578 |
| token-weighted mean | 0.6359 | 0.5052 | 0.6707 | 0.5255 |
| unweighted mean | 0.6349 | 0.5155 | 0.6711 | 0.5402 |

The JSON additionally records every validation-grid result, samples-F1, subset accuracy, and per-label precision/recall/F1/support.

## 20 Newsgroups clustering

Exactly 18,846 canonical rows (train followed by test) were used for every method with identical fixed KMeans parameters.

| representation | NMI | ARI |
|---|---:|---:|
| recursive 424242 (candidate) | 0.6133 | 0.4644 |
| recursive 271828 (final) | 0.6133 | 0.4640 |
| recursive 314159 (final) | 0.6134 | 0.4645 |
| token-weighted mean | 0.5974 | 0.4636 |
| unweighted mean | 0.6144 | 0.4722 |

## Coverage and invariants

No rows were dropped. ECtHR-A used 9,000/1,000/1,000 train/validation/test rows. 20 Newsgroups used 11,314/7,532 train/test rows, including canonical zero-token cleaned rows represented explicitly by one CLS/SEP-only empty-sequence leaf.

Direct-teacher task metrics were intentionally not produced: 344 of 11,000 ECtHR-A rows and 75 of 18,846 20 Newsgroups rows exceeded the pinned teacher's 8,190-raw-token direct context. Valid direct vectors and exact coverage remain in the cache, but subset-only evaluation would violate the identical-sample requirement.

Every non-direct method consumed the same cache tensors. Cache tensor hashes, artifact hashes, dataset revisions/fingerprints, teacher provenance, checkpoint paths/hashes/roles, software versions, and evaluator/protocol hashes are recorded in the machine-readable manifests/results.

## Runtime

Successful cache construction: 1,825.09 s (30.42 min). Evaluation: 290.69 s (4.84 min), including 135.10 s classification and approximately 18.13 s total recursive inference across all three checkpoints. Successful end-to-end runtime was 2,115.78 s (35.26 min) on one NVIDIA GeForce RTX 4060 Ti. The preparation log also preserves an earlier failed fail-closed attempt that terminated on the first canonical empty 20 Newsgroups row; no partial cache survived because preparation was atomic.
