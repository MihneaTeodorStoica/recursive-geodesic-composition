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
