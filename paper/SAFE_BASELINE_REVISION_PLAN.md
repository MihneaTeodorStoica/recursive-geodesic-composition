# Revision plan: matched safe-arm baselines

## Status: improvement 1 completed

Implemented on `revision/safe-arm-baselines`. The three final checkpoints were
downloaded and independently hash-verified; the safe cache was regenerated
without requiring files from the author. All 4,000 document identities and
published prefix-token totals match. No merger was retrained.

See [the completed report](../results/safe_baselines_v1/REPORT.md),
[the audit](../results/safe_baselines_v1/AUDIT.json), and
[reproduction instructions](../reproducibility/protocols/safe_baselines_v1/README.md).
The manuscript now includes the matched comparison, paired intervals,
calibration, costs, and reproduction limitations. Later improvements have not
been started. The steps below retain the original plan and its initial inventory.

## Goal and scope

Address the review's first priority: **does the learned merger improve on a
normalized average on its own teacher-alignment objective?**

Evaluate five representations on the same 4,000 official-v6 safe prefixes,
using the existing three final checkpoints. This requires inference and
representation-source classifier fitting, **not merger/decoder retraining**.
It does not require a new evaluation population or downstream labels.

This document records the plan established before the new scores were computed;
results are reported separately in the artifacts linked above. Recursive-tree
tests and reconstruction calibration remain separate follow-up experiments.

## 1. Recover and verify the minimum inputs

At initial inspection, the checkout contained implementation, configurations,
published summaries, and hashes, but **no `.pt` checkpoints or embedding caches**.
The release downloads and regeneration have since resolved those local gaps.

| Input | At initial inspection | Requirement for this revision |
| --- | --- | --- |
| Model implementation and fixed classifier recipe | Yes | Reuse the existing definitions. |
| Official results, population-order hashes, split hashes, and cache hashes | Yes | Verify identity and compare the recomputed learned row with the published result. |
| Four original official `safe.pt` files | No | Required: ordered leaves, token counts, direct teacher targets, document IDs, and corpus metadata. |
| Original cache `CACHE_MANIFEST.json` and `COMPLETE` | No | Recover with the tensors for provenance verification. |
| Three final `autoencoder_seed*.pt` checkpoints | Not locally; verified in `v1-paper` release | Required for learned reference/final-parent inference and timing; downloadable without recovering private experiment files. |
| TRAIN embeddings | No | Not needed if shuffled pairings provide the calibration diagnostic. |
| Source checkpoints and fitted reconstruction probes | No | Not needed for this targeted experiment. |
| Full-document leaf shards or raw document text | No | Not needed if the original safe tensors are recovered. |

### Preferred recovery

Recover these files from the original experiment machine or backup:

```text
cache_kary4_strict_autoencoder_official_test_v6/
  CACHE_MANIFEST.json
  COMPLETE
  fineweb_edu/safe.pt
  fineweb/safe.pt
  openwebtext/safe.pt
  wikipedia_en_20231101/safe.pt
```

Each safe row should include `leaves` (`[k, 768]`, `k` in 2..4), `lengths`,
`teacher_target` (`[768]`), `id`, `corpus`, and the original prefix metadata.
Use these original safe leaves, not the lower-precision full-document shards.

Obtain the final checkpoints for seeds **424242, 271828, and 314159**. Their
filenames, SHA-256 digests, and release location are in
[`../reproducibility/ARTIFACT_MANIFEST.json`](../reproducibility/ARTIFACT_MANIFEST.json).
They total approximately 1.48 GB. The manifest records approximately 77.2 MB
for the **entire** original official cache; only its safe subset is needed here.

**Release check:** GitHub's API, accessed through the configured GitHub CLI,
confirms that `v1-paper` is the latest release. It contains three final
checkpoints, three source checkpoints, six fitted probes, the artifact manifest,
and checksums. The downloaded manifest and checksum file are byte-identical to
the repository copies; all 12 binary assets' advertised sizes and SHA-256
digests match the manifest. Subsequently, all three required final-checkpoint
binaries were downloaded and independently hashed for the completed experiment;
source checkpoints and probes were not needed.

The release notes and manifest explicitly exclude both the official and TRAIN
caches. Thus the release resolves checkpoint availability, **not the missing
safe-arm input/teacher tensors**. For this revision, the latter were regenerated
from pinned upstream inputs into `cache_safe_baselines_v1/`.

Verify checkpoint and safe-tensor hashes against the repository records. Check
1,000 rows per corpus, pooled/per-corpus document order, dimensions, finite and
nonzero vectors, and exact positive token counts. Expected active-child counts
are **2,707 / 687 / 606** for two / three / four children.

### If the original cache is unavailable

Regenerate the same population from the four pinned corpus revisions and the
pinned `Alibaba-NLP/gte-modernbert-base` model/tokenizer in
[`../configs/official_v6.yaml`](../configs/official_v6.yaml). This needs network
access or external local copies, teacher inference, and additional compute;
it is not an analysis of the checked-in JSON alone.

Use a new, additive cache namespace and preserve the published selection,
tokenization, prefix/chunk rules, masked-mean pooling, and normalization.
Include direct teacher targets for the exact prefixes. Verify population-order
hashes and available token/provenance checks before computing comparisons.
Record regenerated tensor hashes and environment separately: matching document
identity is not proof of bitwise-identical embeddings. Recompute **all** methods
on the regenerated tensors rather than mixing new baselines with old learned
summary scores.

Important: `scripts/v1_k4_pooling.py::prepare_official` creates a
**reconstruction-only** cache without `teacher_target`. It is not a complete
fallback for this experiment. The original official preparation code does
produce targets, but its entry point also enforces full-arm and model/probe
provenance requirements that are unnecessary for a dedicated safe-only study.
Do not weaken or overwrite those frozen protocols; add a separate preparation
entry point if regeneration is needed.

## 2. Freeze an additive analysis protocol

Create a separate configuration, evaluator, and results directory, for example:

```text
configs/safe_baselines_v1.yaml
scripts/evaluate_safe_baselines.py
tests/test_safe_baselines.py
results/safe_baselines_v1/
```

Before computing new comparisons, fix representations, contrasts, classifier
settings, bootstrap seeds, calibration permutations, and timing procedure.
Treat this as a **post-review analysis of an already-used held-out population**,
not a new untouched or originally predeclared official test. Publish every
specified comparison, regardless of direction. Leave frozen official results,
configs, evaluators, and one-shot markers unchanged.

Implement a safe-only loader with explicit integrity checks. The existing
`load_cache()` in the official evaluator requires the full-arm layout; do not
invoke the complete official evaluation just to obtain these five rows.

## 3. Generate the five representations

For active normalized children `x_i` and token counts `n_i`:

| Representation | Definition | Interpretation |
| --- | --- | --- |
| First chunk | `x_1` | Simple truncation-style reference. |
| Unweighted mean | `normalize(sum_i x_i)` | Ordinary flat pooling. |
| Token-weighted mean | `normalize(sum_i n_i * x_i)` | Structural token-mass prior. |
| Learned spherical reference | `aux["anchor"]` | Learned salience before the final correction. |
| Final learned parent | Output of `model.merge(...)` | Complete merger. |

Use all active original leaves, mask padded slots, and normalize each pooled
sum once. Do not replace flat pooling with recursively normalized pooling.
Run the two learned representations for all three checkpoints in evaluation
mode with gradients disabled. The three non-learned representations are
seed-independent and should not be counted as three independent runs.

`GeodesicMerge.merge(..., return_aux=True)` already returns the reference, so
both learned outputs can be collected from the same forward pass without
changing checkpoint weights. The reference is a **learned diagnostic ablation**,
not a zero-cost baseline or an untrained model.

## 4. Compute matched metrics

### Teacher fidelity: primary comparison

- Compute per-document cosine with the exact matching teacher target.
- Report pooled and per-corpus means, plus descriptive results by active arity.
- Report paired differences: final minus weighted mean, final minus unweighted
  mean, reference minus weighted mean, and final minus reference.
- Use 2,000 document-bootstrap resamples, shared across representations and
  checkpoints. For the three-checkpoint mean, average checkpoint scores within
  each document before summarizing/resampling; do not treat 12,000 outputs as
  12,000 independent documents.
- Retain per-checkpoint results, paired 95% intervals, the fraction of documents
  improved, and lower-tail/median/upper-tail differences. Intervals condition on
  these checkpoints, not an unrestricted training-seed population.

### Representation-source discrimination: same recipe

Reuse the pure split/classifier/bootstrap helpers in
`scripts/official_v6/evaluate_official_test_v6_recovery.py`, with
`configs/evaluator_attacks_v1.yaml`:

- Grouped 50:50 split, split and classifier seeds 1729.
- Standardized logistic regression and standardized RBF SVM with unchanged
  hyperparameters; fit scalers only on classifier-training vectors.
- Fit fresh classifiers for **each representation**. Reusing a classifier
  trained only to distinguish the final parent would answer a different question.
- Use identical document memberships across methods within each scope. Verify
  split hashes against the published results. Corpus-specific splits are their
  own fixed splits, not necessarily restrictions of the pooled split.
- Retain both raw and orientation-safe AUCs and identify the stronger family.
  Preserve the original conditional bootstrap interpretation: fixed fitted
  classifiers and observed family choices, no within-bootstrap reselection.
- Report each learned checkpoint and the shared-document three-checkpoint
  aggregate. Compute each deterministic pooling baseline once.

Do not make crossing 0.60 the conclusion. Compare alignment and detectability
as distinct properties; neither establishes downstream utility.

### Cheap calibration without TRAIN data

Include fixed-seed shuffled document–teacher pairings as a diagnostic of the
absolute cosine scale. Use predetermined within-corpus permutations that avoid
self-pairings, reuse them across methods/checkpoints, and distinguish them from
competitive baselines. A TRAIN-derived constant centroid can be added later if
TRAIN embeddings are recovered; never estimate a TRAIN centroid from test data.

### Comparable inference cost

Benchmark composition from already-computed leaves on one recorded machine.
Use the same device, dtype, batch size (64 is the existing safe-root setting),
input population, and timing boundary; include warm-up, repeated measurements,
and CUDA synchronization where applicable. Report hardware, software, and
per-document time or throughput. Exclude teacher encoding, classifier fitting,
decoder evaluation, and cache loading from **all** composition-only timings.

For reference-only timing, an early-exit helper must perform contextualization,
salience, and reference construction and match `aux["anchor"]` numerically.
Alternatively label its measured cost as **full-forward diagnostic extraction**.
Do not time retrieving a cached reference and call that its inference cost.
Do not compare these new measurements directly with the old mixed-overhead
throughput numbers.

## 5. Validate and save an auditable result

Before interpreting outcomes:

- Test masking, variable arity, normalization, token weighting, nonfinite/zero
  input handling, and reference equality with the existing model output.
- Test that document pairs cannot cross classifier splits and that paired
  bootstrap samples are shared across methods/checkpoints.
- Recompute the final-model teacher cosine and AUC alongside the baselines.
  Declare numerical comparison tolerances before running; investigate and
  report discrepancies rather than silently substituting published means.
- Verify checkpoints and frozen official result hashes remain unchanged.

Save a manifest, full-precision metrics, per-document cosines/differences,
classifier split memberships and held-out scores, timing records, and a
human-readable report. Preserve generated vectors as local hashed artifacts
(or provide their reproducible generation path) so later diagnostics do not
require another inference pass. No new reconstruction probes are needed.

## 6. Revise the manuscript after results exist

In [`main.tex`](main.tex):

1. **Evaluation:** define the five representations and the paired comparison;
   explicitly identify the additive post-review protocol.
2. **Safe-arm results:** lead with a matched table of representation, teacher
   cosine, differences versus both pooling baselines, source-detection AUC, and
   composition cost. Put detailed per-seed/stratified values in the appendix
   when necessary for readability.
3. **Abstract, discussion, and conclusion:** replace the missing-baseline
   limitation with the measured result and its uncertainty. State clearly
   whether learning improves, fails to improve, or shows mixed behavior across
   the prespecified comparisons. Do not declare equivalence from a nonsignificant
   difference or extrapolate a fidelity win to downstream usefulness.
4. **Reproducibility appendix:** document hashes, cache recovery/regeneration,
   fixed classifier settings, timing boundaries, and artifact paths.

The revision is complete when readers can directly see **what learning adds
beyond pooling, what the final correction adds beyond the learned reference,
and what those differences cost**—even if the answer is no demonstrated gain.
