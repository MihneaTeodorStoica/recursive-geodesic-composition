# Post-review matched safe-arm baselines

This is improvement 1 only: compare first-chunk, flat unweighted pooling, flat
token-weighted pooling, the learned spherical reference, and the final learned
parent on the published 4,000 safe-prefix documents. It is **not** a new untouched
official test, a merger-training run, a recursive-depth test, or a reconstruction
experiment. The original official results, evaluators, model implementation,
and checkpoints are unchanged.

## Reproduce

From the repository root, use the pinned Python 3.12 environment described in
the main README. Download only the three final checkpoints:

```bash
mkdir -p artifacts/v1-paper
# Obtain the exact manifest-listed weights through the anonymous submission discussion.
# Supply a flat local directory containing the exact .pt assets, then:
python supplement.py materialize /path/to/weights --role final_clean_v6_checkpoint --mode hardlink

# Public pinned upstream datasets/model are required only for cache regeneration.
.venv/bin/python scripts/prepare_safe_baselines.py --device cuda

# Merger inference, fixed source classifiers, paired intervals, and reporting.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/evaluate_safe_baselines.py all --device cuda

.venv/bin/python scripts/audit_safe_baselines.py
.venv/bin/python -m pytest -q
```

Existing downloads need not be repeated. `prepare` verifies and reuses a
complete cache; partial preparation resumes only under the same configuration
and preparation source hashes. Inference can be called separately with `infer`,
followed by `evaluate`. Fitted classifier results are saved per method/scope
and reused on an interrupted analysis only if their input/protocol hashes match.
A completed analysis is create-only; `report` regenerates presentation files
without refitting or rerunning the experiment.

Configuration: [`../../../configs/safe_baselines_v1.yaml`](../../../configs/safe_baselines_v1.yaml).
Results: [`../../../results/safe_baselines_v1/REPORT.md`](../../../results/safe_baselines_v1/REPORT.md).
The three checkpoints, regenerated cache, and `vectors.pt` are local ignored
binary artifacts; JSON scores, split memberships, classifier predictions,
provenance, and checksums are retained with the analysis.

## Matching and precision

- The pinned source revisions, scan `[162000,192000)`, first-1,000 eligible
  selection per corpus, tokenizer, 1,025–4,096-token prefixes, and 1,024-token
  chunks come from the original official configuration.
- Document-order hashes and prefix-token totals match the published records.
  Active-child counts must be 2,707 two-child, 687 three-child, and 606
  four-child documents.
- Documents are tokenized once, without special tokens or truncation, then
  sliced as token-ID sequences. Counts exclude special tokens; there is no
  decode/re-tokenize step.
- The original `FrozenTeacher` loader is reused unchanged. The pinned teacher
  loads FP16 parameters; its masked-mean-pooled, normalized vectors are stored
  in FP32. The merger and all pooling calculations use FP32. TF32 is disabled.
- Regeneration embeds only the safe leaves, up to four per document, rather
  than all full-document leaves in batches up to 16. Thus **document identity
  is verified but original embedding byte identity is not claimed**. All new
  comparisons use the same regenerated inputs and freshly recomputed parents.
- The original teacher's documented first-token interface is not substituted
  for the masked-mean representation used during merger training.

An initial new preflight incorrectly conflated teacher parameter precision with
embedding-storage precision. It stopped before producing a safe embedding
cache or any comparison score. The correction preserved the original loader
rather than forcing FP32 weights; the failed local attempt was archived.
See [`PREPARATION_RECOVERY.json`](PREPARATION_RECOVERY.json). No outcome-driven
model, population, or classifier change was made.

## Metrics and conditioning

Teacher cosine is paired by document. Report the final parent against both
pooling baselines, the learned reference against weighted pooling, and the
parent against its reference. A 2,000-replicate bootstrap shares document draws
across all methods and checkpoints. Three-checkpoint fidelity is the mean of
checkpoint scores for each document, not a fourth model or 12,000 independent
documents. Per-checkpoint, per-corpus, and active-arity summaries are retained.

Source discrimination reuses the frozen evaluator's standardized logistic/RBF
functions and settings. Each representation receives its own fitted classifiers,
but every method uses the same document split within each scope. Both teacher
and generated vectors from a document stay in the same partition. Split hashes
are verified against the original pooled and separately fitted per-corpus
attacks. Folded AUC is `max(AUC, 1-AUC)`; the stronger family is selected once
on the observed scores. Intervals condition on that choice, the fitted
classifiers, split, and checkpoints; attackers are not refitted or reselected
inside a bootstrap replicate.

Calibration uses 100 fixed within-corpus teacher derangements without
self-pairings, shared across methods/checkpoints. This requires no TRAIN cache.
Its permutation range is descriptive, **not** a sampling confidence interval
or a downstream retrieval benchmark.

## Timing

Measure device-resident leaves to device-resident output, with the same arity
batches of up to 64 documents, three warm-up passes and 20 measured passes.
Synchronize CUDA at timing boundaries. Report the median pass time divided by
4,000, then average checkpoint medians for learned methods. This is amortized
batched composition time, **not batch-one latency**. Exclude teacher encoding,
cache I/O, host/device transfers, decoding, and classifier fitting from all
methods. Hardware and software are recorded in `INFERENCE_MANIFEST.json`.

The learned reference is extracted from the same complete forward call as the
parent; their reported timing is the same measured operation. This does not
estimate the cost of an optimized reference-only implementation or treat the
reference as a free pooling baseline. These timings must not be combined with
the old mixed-overhead official or scaling-study throughput figures.

## Validation and scope

The audit checks population and artifact identity, independently recomputes
cosines using PyTorch double precision, recomputes paired fidelity intervals,
checks stored classifier scores and AUCs, verifies split pairing and three-seed
aggregation, and checks unchanged checkpoint/official-result hashes. It does
not independently refit classifiers or recompute their bootstrap intervals.

Recomputed learned results are compared with the published values using
pre-specified absolute diagnostic tolerances of `1e-5` for mean cosine and
`5e-4` for AUC. Deviations are reported, not used for selection or suppression.
The historical publication checksums in `reproducibility/hashes.txt` describe
the original snapshot; revision artifacts have their own hash manifest.
