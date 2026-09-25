# Matched continuation comparison (post-review improvement 4)

Completed: [results](../../../results/continuation_comparison_v1/REPORT.md),
[independent audit](../../../results/continuation_comparison_v1/AUDIT.json), and
[validation record](VALIDATION.md). The primary plan below was established
before the new evaluation; post-result checks and layout changes are identified
separately at the end.

## Question and scope frozen before new evaluation

What changes between each released source merger and its fixed clean-v6 final
checkpoint? Compare teacher cosine and representation-source AUC on the same
4,000 regenerated safe prefixes, and copy-calibrated reconstruction by the
matched released source/candidate TRAIN probes on the same 606 exact-four
prefixes. Reuse final outputs and classifier scores from improvements 1 and 3.
This is additive analysis of previously evaluated documents, **not an untouched
confirmatory test**, an objective-isolating ablation, or training from scratch.

The final checkpoints identify the exact source bytes in `staged_training`.
The continuation discarded source decoder weights and used a freshly initialized
decoder, 300 decoder-only updates, then 100 joint updates (encoder LR scale
0.02). The source is already trained. This comparison cannot isolate the
reconstruction objective, distribution losses, curriculum or training budget.

## Fixed choices

- Configuration: `configs/continuation_comparison_v1.yaml`. Freeze it and the
  evaluator/statistical/rendering/audit implementation hashes in
  `ANALYSIS_PROTOCOL.json` **before new inference**.
- Verify release SHA-256, size, role and seed for all three source/final mergers
  and six probes. Confirm exact source lineage, identical inference architecture
  and the original recorded change in all 113 merger tensors.
- No teacher encoding, merger/decoder/probe updates, or new probe selection.
  Infer source parents and source-probe predictions only. Final parent/probe
  tensors must remain bitwise equal to the preceding studies.
- Probe metadata must match across each source/candidate pair: fresh-v6 TRAIN
  role, selected training/floor groups and deterministic minibatch schedule.
  Matched initialization/architecture/fitting are documented in the original
  frozen protocol and `scripts/final_v6/freeze_training_and_probes.py`; no claim
  of independently reproducing probe training is made. These are not the
  historical separately fitted matched-pooling probes.
- Teacher cosine: all 4,000 documents; per-corpus and active-arity (2/3/4)
  descriptive summaries. Source/final output cosine and weighted-pooling
  contrasts are secondary diagnostics.
- Source AUC: unchanged standardized logistic and RBF classifiers, same
  hyperparameters and document-grouped 50:50 splits. Fit both families for each
  source checkpoint and scope (30 fits). Reuse all 30 final classifier score
  bundles with their previously committed hashes; do not apply final-trained
  classifiers to source outputs. Verify split hashes against the published
  protocol. The pooled AUC has 2,000 held-out document pairs; corpus AUCs each
  have 500, using their own frozen splits.
- Reconstruction: ordered probe cosine, own-parent-copy cosine, probe gain
  over that copy, and ordered-minus-all-24-target-permutations gap. Compare
  matched source/final endpoints and the change in gain. Slots have equal
  weight. Hold predictions fixed when permuting targets; no inference reruns
  or permutation-based selection. These are local direct-parent diagnostics.
- Primary contrasts: final minus source in teacher cosine, orientation-safe
  source AUC, and probe gain over own-parent copy. Report every checkpoint,
  corpus, arity and direction, not just improvements or threshold crossings.
- 2,000 shared document resamples within each metric/scope. Average checkpoint
  scalar scores within each document before their bootstrap; for AUC instead
  compute each checkpoint AUC and average those AUCs within each resample.
  Both teacher/generated observations share each document's multiplicity.
  Fix observed strongest classifier families; no classifier refits or family
  reselection during bootstrap. Recompute the orientation-safe convention
  within each resample. Intervals condition on fixed checkpoints, probes,
  classifiers and observed family choices, not a new training-seed population;
  no simultaneous/multiple-testing guarantee is claimed.

## Commands

Requires the verified artifacts and preceding ignored vector/cache files from
improvements 1–3. Obtain `source_seed{seed}.pt` and
`source_parent_only_probe_seed{seed}.pt` from release `v1-paper` into
`artifacts/v1-paper/`; check `reproducibility/ARTIFACT_MANIFEST.json`. Downloaded
model assets and caches remain ignored, never silently substituted.

```bash
.venv/bin/python scripts/evaluate_continuation_comparison.py all --device cuda
.venv/bin/python scripts/audit_continuation_comparison.py --device cuda
# Supplementary post-result numerical check; create-only, not a primary rerun:
.venv/bin/python scripts/check_continuation_numerics.py --device cuda
# Frozen rendering plus documented layout-only polish:
.venv/bin/python scripts/publish_continuation_comparison.py
.venv/bin/python -m pytest -q
artifacts/tools/tectonic --keep-logs --keep-intermediates paper/main.tex
```

Inference and completed results are create-only. If classifier execution is
interrupted before `RESULTS.json` exists, `analyse` may resume using already
saved, hash/configuration-matched classifier outputs; it never selectively
refits a completed classifier. Report rendering and independent audit may be
repeated. Preserve any failures; do not change tolerances after outcomes.

As with the preceding studies, an independent cache regeneration is numerical
replication, not promised byte identity. Use a separate workspace and explicitly
freeze any different input fingerprints. Do not bypass this study's integrity
checks or mix regenerated source scores with historical final scores.

## Audit plan

Independently replay every source merger and probe with batch size 17 (versus
64), using mixed-arity document-order packing instead of arity grouping and
replayed source parents as probe inputs. Fixed maximum absolute tolerance:
`2e-6`. Recompute cosines independently in PyTorch double precision. Recompute
all paired scalar/AUC summaries. Independently compare the weighted tie-aware
Mann–Whitney AUC implementation against scikit-learn on observed points and
32 shared bootstrap resamples per stage/checkpoint/scope, in addition to
synthetic tie/weight/orientation/pairing tests. Verify unchanged final outputs,
source/final probes, classifier splits, all checkpoint bytes and prior results.

Public scalar scores, cross-slot cosine matrices, classifier scores/splits,
protocol, lineage, summaries and independent audit are saved in
`results/continuation_comparison_v1/`. Full tensors remain ignored and hashed.

## Explicitly post-result checks and presentation

The observed pooled teacher-cosine change is only `8.465565854e-7`. This
motivated a supplementary replay of **both** endpoints with mixed-arity
batch size 17, independently recomputing teacher cosines in double precision.
Its separate implementation and `NUMERICAL_CHECK_PROTOCOL.json` were recorded
before that additional replay. The pooled contrast shifts by only
`+1.593617995e-11`; every numerical shift is retained, and no primary score,
interval, checkpoint, document or tolerance is changed. This is a post-result
numerical validation, not a second selected evaluation.

The original renderer remains byte-identical to the pre-inference freeze.
`publish_continuation_comparison.py` applies only layout polish: extra x-axis
label padding avoids overlap with scientific-notation offsets, and table
changes/CIs are displayed in units of `1e-6` so the tiny teacher interval does
not round to identical bounds. Endpoint columns and the raw Markdown report
remain unscaled. `PRESENTATION.json` records the wrapper hash and unchanged
primary-result digest. Running the plain frozen renderer restores its original
formatting; use the publication wrapper to reproduce the manuscript layout.

The substantive finding is that nearly all of the observed safe-arm advantage
over pooling predates the last continuation stage. Teacher cosine increases by
less than `1e-6`, and source AUC decreases by approximately `0.000533`. The
probe's own-copy-relative gain becomes less negative, but its raw ordered
cosine falls and both endpoints still underperform parent copy. None of these
endpoint comparisons isolates a loss or establishes a from-scratch result.
