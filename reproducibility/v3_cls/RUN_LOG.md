# V3 CLS run log (append-only)

## 2026-09-02T00:00:00Z initialization
- Created branch `v3-paper-cls` from `origin/main` commit `c6b3f28830e7965533e1d036c2a749836b7848b6` (V1 publication base).
- Configured git author/committer to repository owner's noreply identity: Mihnea-Teodor Stoica <MihneaTeodorStoica@users.noreply.github.com>.
- Historical branches/artifacts are read-only inputs for audit only; V3 namespaces are isolated under `*/v3_cls/`.

## 2026-09-02T00:10:00Z teacher/range guard implementation
- Ported explicit `FrozenTeacher(pooling=...)` support while preserving V1 default `masked_mean`.
- Added V3 hard guards in `train.py`: V3 requires pinned GTE ModernBERT revision, `teacher_pooling: cls`, `v3_cls` cache/output namespaces, and no learned initialization/continuation.
- Added V3 cache sidecar checks requiring CLS teacher provenance before tensors can be loaded.
- Added isolated V3 configs, TRAIN-only cache preparer, source-range auditor, CLS reference validator, and guard tests.
- Ran: `PYTHONPATH=src python scripts/v3_cls/audit_source_ranges.py`; result `protected_highest_exclusive=296000`, V3 allocation `[296000,348000)`, non-overlap true.
- Ran: `PYTHONPATH=src pytest -q tests/test_v3_cls_guards.py tests/test_geodesic_parameterization.py`; result 10 passed.
- Ran: `PYTHONPATH=src python scripts/v3_cls/validate_cls_teacher.py --device cuda`; result passed with max_abs_diff=0.0 against direct documented CLS usage under identical batching.

## 2026-09-02T00:25:00Z pilot cache/training failure and engineering fix
- Prepared V3 CLS pilot TRAIN cache from the frozen V3 TRAIN start `[296000,296016)` per corpus; this marks those TRAIN rows accessed for pilot engineering.
- First pilot training attempt failed before optimization: inherited warmup_steps=50 exceeded pilot steps=20. This is a pilot-config engineering error, not a scientific result. Fixed pilot config only by setting warmup_steps=5; final configs remain at steps=1200/warmup=50.
- Second pilot training attempt failed before optimization because `decoder_only_require_full_arity_rows` is only valid for a frozen strict autoencoder stage, while V3 intentionally trains merger from step 0. Fixed all V3 configs to set `decoder_only_require_full_arity_rows: false`; this matches the predeclared fresh-from-random no-freeze protocol.
- Pilot training completed for `configs/v3_cls/pilot_seed424242.yaml` using the tiny pilot cache. Final step 20 was finite: loss=0.061259, teacher=0.039314, grad_norm=0.629885, geodesic_saturation_fraction=0.0. This is an engineering health check only; decoder objectives had not started.

## 2026-09-02T08:25:00Z full TRAIN cache preparation engineering recovery
- Freed workspace capacity by moving (not deleting) ignored/untracked V2 local artifacts to `/root/v2_local_archive`; historical files remain intact outside the V3 worktree.
- First full cache command failed before data/model access because the namespace placeholder `artifacts/v3_cls/cache_train_cls/` was an empty directory created during initial scaffolding. Verified it contained no files, removed only that empty placeholder, and retried the create-only preparation unchanged.
- Full create-only V3 TRAIN CLS cache completed over `[296000,308000)` per corpus: usable augmented rows fineweb_edu=6898, fineweb=3760, openwebtext=8228, wikipedia=6282; size 319 MiB. Teacher provenance in every sidecar is pinned normalized CLS.
- Predeclared `TRAIN_PROTOCOL_AMENDMENT_V1.md`: one bounded 80-step full-architecture TRAIN diagnostic, followed on pass by identical fresh 2400-step staged single-run protocol for all final seeds.
- Full-architecture diagnostic passed all predeclared criteria: finite=true; teacher loss first20 mean=0.027238, last20=0.024386; grad_norm max=0.7913; saturation max=0.0. Amendment V1 candidate protocol is accepted for the first final seed candidate.
- First final training launch failed before Python execution because `/usr/bin/time` is not installed. Retried unchanged using shell timestamps; no model/data semantics changed.
- Three Amendment-V1 candidate models completed at fixed step 2400 in 349s, 345s, and 345s (seeds 424242, 271828, 314159). All histories were finite with zero geodesic saturation.
- First TRAIN-only probe-freeze attempt failed before writing a probe due to indexing a CPU root tensor with CUDA indices. Preserved the partial namespace as `artifacts/v3_cls/predevelopment_freeze.failed_device_indexing`, fixed only tensor-device indexing, and retried with identical rows, schedule, objectives, and seeds.

## 2026-09-02T09:10:00Z development round 1 and bounded amendment V2
- Marked V3 development role accessed before opening data. Prepared first 1000 source-order eligible 1025+ token documents/corpus from `[308000,317000)`; actual maximum examined source index was 313998. Safe/full arms share tensor-identical leaf embeddings.
- Evaluated all three Amendment-V1 candidates. Pooled cosine: 0.987137/0.987087/0.987172. Strongest logistic AUC: 0.695108/0.694672/0.697745; RBF AUC: 0.597706/0.595108/0.597279. All probe floors passed; full recursion maximum depth=3, maximum leaves=46, no geodesic saturation.
- Added evaluator-only distribution diagnostics on the already-spent round-1 cache. Variance ratios were nearly one, but coherent mean-vector L2 was ~0.0116; logistic materially exceeded RBF. Predeclared exactly one late linear-adversarial fresh candidate and mechanical selection rule in `TRAIN_PROTOCOL_AMENDMENT_V2.md`; no covariance objective or sweep.
- First round-2 baseline evaluator invocation failed before opening cache/checkpoint because the requested output label contained forbidden historical namespace token `v1`. Renamed only the output directory label to `baseline_amendment1`; scientific inputs and semantics unchanged.
- Development round 2 accessed only fresh `[314000,317000)` and evaluated 400 eligible documents/corpus. Adversarial candidate met the predeclared cross-round selection rule (cosine=0.986920, strongest AUC=0.651691, rank=121.52, retention healthy, zero saturation).
- Same-population Amendment-V1 baseline AUC was 0.650813, slightly better than adversarial, so causal benefit is unsupported. Per the frozen rule, selected V2 anyway and precommitted no further search; documented this important negative result in `AMENDMENT_V2_SELECTION_DECISION.md`.
