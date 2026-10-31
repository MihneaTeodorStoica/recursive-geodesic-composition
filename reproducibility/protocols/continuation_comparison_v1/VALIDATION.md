# Completed validation: continuation comparison

## Scope and outcome

- All three released source/final checkpoint pairs verified against the release
  manifest by SHA-256, size, seed and role. Each final checkpoint identifies
  the exact source digest and path in `staged_training`. Architecture signatures
  match; all **113/113** merger tensors changed in the original continuation.
- All six released TRAIN probe artifacts verified. Source/candidate selection,
  floor-group and minibatch-schedule metadata match. No probe was retrained.
- **4,000** safe prefixes and **606** exact-four prefixes retained, with identical
  document order, teacher/child targets and final tensors from preceding studies.
- **30** new fixed source-classifier fits; **zero** final-classifier refits.
  All 30 previously fitted final-classifier score bundles remain unchanged.
  Both families are retained; logistic regression is strongest for every
  stage/checkpoint/scope. Identical source/final classifier split memberships.
- No merger/decoder/probe updates, teacher encoding or alternate checkpoint
  selection. No primary results or tolerances changed; no documents or completed
  primary runs discarded.

| Pooled measurement | Source | Final | Paired final-minus-source mean |
| --- | ---: | ---: | ---: |
| Teacher cosine | 0.985044633 | 0.985045479 | +0.000000847 |
| Source AUC | 0.584827167 | 0.584293875 | -0.000533292 |
| Parent-copy child cosine | 0.914068899 | 0.914021511 | -0.000047389 |
| Probe child cosine | 0.908114434 | 0.908105018 | -0.000009416 |
| Probe gain over own-parent copy | -0.005954465 | -0.005916492 | +0.000037973 |
| Probe order gap | 0.000910307 | 0.000908435 | -0.000001872 |

Teacher-cosine change CI: `[6.101722265e-7, 1.081134565e-6]`.
AUC change CI: `[-0.000675928125, -0.000392327083]`.
Probe gain change CI: `[0.000035858520, 0.000040216195]`.
These are 2,000-resample document-paired conditional intervals, not simultaneous
coverage or uncertainty over newly trained models. Raw probe cosine declines;
the copy-relative gain becomes less negative because copy cosine declines more.

**99.96%** of the observed final teacher-cosine gain over weighted pooling is
already present in the source. Neither this endpoint arithmetic nor the small
changes isolates the decoder objective or establishes a from-scratch result.
Corpus and arity exceptions are retained: Wikipedia AUC increases slightly,
and the three-child/Wikipedia mean teacher changes are slightly negative;
these checkpoint-mean intervals span zero.

## Independent checks

- Replay all source parents using mixed-arity document-order batches of **17**,
  rather than arity-grouped batches of 64. Replay all source probes from those
  independently recomputed parents. Maximum absolute parent/probe component
  difference: **5.960464477539063e-8**, below the fixed **2e-6** tolerance.
- Recompute teacher and cross-slot child cosines independently in PyTorch double
  precision; verify copied-parent baselines, order gaps, paired contrasts and
  every scalar bootstrap summary.
- Recompute all paired AUC intervals. Synthetic tests cover ties, reversed
  orientation, invalid weights, repeated/zero-weight documents, row permutation,
  fixed pairing and mean-of-checkpoint-AUC aggregation. Published tests compare
  every per-checkpoint observed AUC against the original scikit-learn output.
- Independently check **32 shared resamples per stage/checkpoint/scope** against
  scikit-learn weighted ROC AUC. Maximum absolute discrepancy:
  **1.1102230246251565e-16**.
- A supplementary, explicitly **post-result** numerical check was motivated by
  the sub-1e-6 teacher change. Its code/protocol were recorded before replaying
  both source and final endpoints with batch size 17. The pooled teacher-change
  shift is **+1.593617994973447e-11**; maximum absolute checkpoint-mean document
  shift is **4.300723110972626e-9**. Largest endpoint vector component difference
  is **8.940696716308594e-8**. All shifts are saved; no primary output or
  predeclared tolerance was replaced. This verifies numerical stability of this
  replay, not original embedding-byte identity or all regeneration uncertainty.
- A separate layout-only publication wrapper leaves the frozen evaluator and
  renderer byte-identical. It pads figure labels and expresses table changes/CIs
  in `1e-6` units, avoiding a spuriously degenerate-looking rounded teacher CI.
  `PRESENTATION.json` records the unchanged primary-result digest.

## Software and manuscript validation

```text
.venv/bin/python -m pytest -q                       133 passed
.venv/bin/python -m compileall -q src scripts tests passed
uv pip check --python .venv/bin/python              96 packages compatible
scripts/audit_continuation_comparison.py            passed
scripts/check_continuation_numerics.py               completed, all shifts retained
tectonic paper/main.tex                            32 pages, no TeX warnings
git diff --check                                   passed
```

Initial layout warnings from long inline numeric intervals were fixed by
rewriting the surrounding sentences, without changing numbers. The published
page was visually checked after figure-label padding/table-unit polishing;
abstract, methods, surrounding reconstruction results and audit appendix were
also inspected. PDF, logs and rendered inspection pages remain local ignored
artifacts. Primary result files are create-only and all prior study outputs
remain unchanged.

## Result identities

- `results/official_v6/results.json`:
  `2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917`
- `results/safe_baselines_v1/RESULTS.json`:
  `f16d23f504d3155dce680243a680d6c26ed2835b46f6aef9100b7fd9c1c57da5`
- `results/recursive_reuse_v1/RESULTS.json`:
  `74944d0960b9311ef8b897ef130474b6a04d463cac4ea6222cb8d96aa6614e29`
- `results/reconstruction_calibration_v1/RESULTS.json`:
  `6f378128f85bcac3ae762c9a931985e3fd5a110d08cc8334bcfef87746bc46c7`
- `results/continuation_comparison_v1/RESULTS.json`:
  `84c151264e962577a35cd12ad47ace1a63852d5abcc8ddcaa1784827d23b16c1`
