# Completed additive review-control validation

Recorded against the new results with configuration `configs/review_completion_v1.yaml`. The original official and four earlier additive result digests are unchanged and explicitly checked by `scripts/review_common.py` and the offline publication tests. All study-specific intervals are conditional; no classifier, probe, method, corpus, or task was dropped for sign.

## End-to-end checks

```text
python scripts/audit_review_completion.py input_order tree_families probe_families decoder_gradient longembed_review
AUDIT PASSED: input_order
AUDIT PASSED: tree_families
AUDIT PASSED: probe_families
AUDIT PASSED: decoder_gradient
Independent retrieval metrics/intervals verified: 2wikimqa
Independent retrieval metrics/intervals verified: qmsum
Independent retrieval metrics/intervals verified: summ_screen_fd
Independent retrieval metrics/intervals verified: narrativeqa
AUDIT PASSED: longembed_review

python -m pytest -q
162 passed
uv pip check --python .venv/bin/python
All installed packages are compatible
python -m compileall -q scripts src tests

git diff --check
(no diagnostics)

cd paper && SOURCE_DATE_EPOCH=1790294400 tectonic -X compile main.tex
35 A4 pages; TeX, BibTeX and rerun passed without warnings; PDF inspected as image
on pages 1, 21, 22, 23, 25 and 31. Two fixed-epoch builds were SHA-256 identical:
b88e0d0d1165359231e75e837c8b197311e2bb14993b13b9b094db8225c33f81.
See the v2-paper-review release PDF asset.
```

The tests and PDF must be rerun after publication-text or manifest edits before tagging; the final release digest in the manifest is authoritative. The absent `latexmk` executable was not misreported as success: compilation uses the available Tectonic binary. A source-only clone does not include ignored GPU caches, so the independent numeric audit needs the verified `v1-paper` weights and regenerated score-vector artifacts in addition to the tracked JSON.

## Independent checks

- All 24 input permutations and both count modes were rescored per document and seed from saved predictions with double-precision dot products; fixed-parent controls cancel to ~1e-18. Maximum alternative input-order replay delta: `1.4901161193847656e-07` vs frozen `2e-6` cap.
- Five binary trees, three segmentations, three 1,024/512/256-token binary/K4 settings, and flat pools retain original document/teacher identities and mass conservation. Every depth/merge count and teacher cosine was recomputed. Maximum replay delta: `8.940696716308594e-08`.
- New probe fits: 800 selected corpus-indexed identities, 640 fit and 160 unused validation, zero overlapping exact evaluation token hashes; 45 new probe checkpoints, their fixed 400-step schedules and shared initializations independently checked. Child cosines were recomputed from predictions. Maximum replay delta: `1.1920928955078125e-07`.
- Decoder intervention: six step-400 models, three shared step-300 warm weights and optimizers, matching per-seed joint minibatches and initial forward losses; original losses held fixed except joint-stage gradient route. 4,000 document-paired teacher scores, 606 raw/copy-relative child scores, five scopes' 30 AUC classifier-score bundles and paired bootstrap intervals were recomputed. See `results/decoder_gradient_v1/AUDIT.json` and `FROZEN_MODELS.json`.
- LongEmbed: SHA-256 for each of 12 pinned corpus/query/qrel files; complete four task candidate sets and query identities; first/weighted/unweighted/context references and three model roots; manual stable-ID nDCG/recall/MRR rankings, double-precision query--document similarities and all fixed positive-document-cluster bootstrap intervals independently verified. Recursive alternate batching replay maximum `8.940696716308594e-08`. Dataset-derived vectors and full scores are in the local per-task results for detailed inspection, with scores committed to Git and vector SHA-256 in the release manifest. The partial-run preservation manifest verifies that the original three scored tasks are byte-identical through the parser recovery.
- The `v2-paper-review` assets were SHA-256 checked against `reproducibility/REVIEW_RELEASE_MANIFEST.json`; `scripts/materialize_review_release.py --verify-only` and `sha256sum -c REVIEW_RELEASE_ASSETS.sha256` both pass on the prepared package. The old `v1-paper` release remains unchanged and is required for historical artifact identities.

## Explicit discrepancies and inferential constraints

Pre-existing unchanged discrepancies remain in the original safe-arm FineWeb AUC and reconstruction-calibration OpenWebText historical tolerance reports. Initial TRAIN target 500/100 was infeasible before any embedding/fitting; recovered 160/40 selection has a separate protocol. The first retrieval attempt failed on FP16 mean summation before scoring; the uniformly FP32 benchmark has a separate source fingerprint. The second attempt failed on Unicode-aware `splitlines()` after scoring three tasks; the repair retained those exact outputs before completing the fourth. Neither failure is erased or described as a successful initial run. CHECK `paper/REVIEW_EVIDENCE_MAP.md` for teacher-objective, downstream, and training-path distinctions.
