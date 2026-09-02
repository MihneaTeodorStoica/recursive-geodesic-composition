# V3 CLS final status report

Status: **not ready for human release review**. This branch currently contains the V3 CLS experiment foundation, provenance, guards, allocation, and pilot smoke only. Full TRAIN cache construction, three final seed trainings, development/model-selection, official one-shot evaluation, downstream evaluations, figures, and release manifest have **not** been run.

## Completed
- Branch `v3-paper-cls` created from V1 publication base commit `c6b3f28830e7965533e1d036c2a749836b7848b6`.
- Git author/committer configured to repository owner identity.
- V1 architecture/training-lineage audit written in `V1_AUDIT.md`; missing pre-final continuation lineage is explicitly identified.
- Explicit teacher pooling implemented. V1-compatible `masked_mean` remains default; V3 guards require `cls`.
- CLS reference validation passed against direct documented `last_hidden_state[:,0]` + L2 normalization: `max_abs_diff=0.0`, `cosine_min=0.9999999403953552` under identical batching.
- Programmatic range audit written and run. Protected highest exclusive index is 296000; V3 allocation is TRAIN `[296000,308000)`, development `[308000,317000)`, buffer `[317000,318000)`, future official `[318000,348000)`.
- Isolated namespaces and configs under `configs/v3_cls/`, `scripts/v3_cls/`, `artifacts/v3_cls/`, `results/v3_cls/`, `reproducibility/v3_cls/`, `figures/v3_cls/`.
- Runtime hard guards prevent V3 from using historical pooling, historical namespaces, or learned initialization/continuation.
- TRAIN-only pilot cache and 20-step tiny pilot completed after documented config engineering fixes.

## Pilot headline
Final pilot step: loss `0.061259`, teacher loss `0.039314`, finite gradients (`grad_norm=0.629885`), geodesic saturation fraction `0.0`. Decoder objectives had not started in this 20-step engineering pilot.

## Not completed
- Full V3 TRAIN cache from all 12k source indices per corpus.
- Three final seed trainings (`424242`, `271828`, `314159`) under frozen 1200-step fresh protocol.
- Development evaluation/model-development phase.
- Pre-official freeze, official one-shot evaluation, official CIs/per-corpus metrics.
- Downstream retrieval/classification/clustering evaluations.
- Publication figures and proposed `v3-paper` release manifest.

## Scientific/audit status
No official V3 range has been accessed. No learned V1/V2/V2.2 artifacts were loaded. No historical branches or release artifacts were modified. The branch is safe to continue from the recorded state, but V3 is not complete.
