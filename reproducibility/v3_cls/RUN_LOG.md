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
