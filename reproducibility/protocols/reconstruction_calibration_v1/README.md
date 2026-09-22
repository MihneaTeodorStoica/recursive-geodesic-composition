# Improvement 3: reconstruction calibration

Completed: [report](../../../results/reconstruction_calibration_v1/REPORT.md),
[audit](../../../results/reconstruction_calibration_v1/AUDIT.json), and
[validation record](VALIDATION.md). The design below was fixed before generating
the new predictions; all results and numerical discrepancies are retained.

## Fixed design before new predictions

This is an additive post-review analysis of the same 606 exactly-four-child
safe prefixes, not a new held-out test. Reuse the original leaf vectors and
the direct learned / flat-pooling parents from improvements 1 and 2. Do not
use the binary recursive roots: this calibrates the earlier local exact-K4
reconstruction diagnostic, not root-to-all-leaf recovery.

For each of the three frozen final checkpoints, evaluate:

1. **Parent copy:** repeat the normalized learned parent in all four slots.
2. **Built-in strict decoder:** one parent vector is its only input.
3. **Released candidate TRAIN-fitted probe:** the corresponding seed-specific
   probe, frozen; one parent vector is its only input.

Also repeat the flat token-weighted and unweighted normalized means in all
four slots, without a fitted decoder. All metrics weight the four children
equally, matching the earlier ordered-child metric, then weight documents
equally. Unweighted pooling is the best shared unit vector under this metric:
`mean_i dot(z,x_i) <= norm(mean_i x_i)` for unit children and any shared unit z.
This does not bound a decoder that produces distinct slot-specific vectors.

The three released candidate probes are not assumed to equal the separately
fitted seed-424242 matched-pooling probes. The latter binaries (including the
two pooling probes) are not release assets. Retain their historical comparison
unchanged; do not combine new copy scores with old decoded scores to estimate
a supposedly matched gain. No new probes or mergers are fitted in this step.

## Ordered versus shuffled target scoring

Preserve inputs and predictions and permute **target-slot assignments within
each document**, not input children or documents. Exhaust all 24 permutations,
including the identity, and separately all nine derangements with no fixed
slots. There is no Monte Carlo permutation selection.

For cross-cosines `C[i,j] = cos(predicted_slot_i, target_slot_j)`, report:

- ordered score `trace(C)/4`;
- all-24 permutation mean `sum(C)/16`;
- nine-derangement mean `sum(off_diagonal(C))/12`;
- ordered minus each shuffled score;
- the best of 24 assignments, and best minus ordered, labeled **oracle**
  diagnostics requiring target labels, not a reconstruction method.

A copied parent must have zero order advantage. Order differences at or below
1e-12 in magnitude are numerical zero; this only removes double-precision
roundoff from algebraically identical scores. A positive order gap need not
prove document-specific order storage: a decoder can exploit positional priors.

## Contrasts and uncertainty

Primary paired contrasts are built-in decoder minus learned-parent copy and
released probe minus learned-parent copy. Also compare both decoded outputs
against the two pooling-copy baselines. Report every contrast and outcome,
including negative gains, per checkpoint, pooled, and by corpus.

Use 2,000 shared document resamples with seed 22092621 plus fixed scope offsets.
Average checkpoint scores within each document before aggregate resampling;
there are 606 document units, not 2,424 independent child observations or 24
permutation observations per document. Report empirical gain/order-gap CDFs.
Intervals condition on these fixed checkpoints and previously fitted probes.

## Geometry and reproduction checks

Both decoder families use `normalize(p + b)` with `norm(b) <= 0.25`, placing
predictions within `asin(0.25)` of their unit parent. Quantify the fraction of
true children outside that radius and the loose oracle cosine upper bound
`cos(max(acos(dot(p,x)) - asin(0.25), 0))`. This bound ignores the tighter
coordinate box and information bottleneck; it is not an achievable score or
proof that the bound caused a particular error. Verify actual decoder movements.

Compare recomputed ordered means with the original official built-in/probe
means as a reproduction diagnostic, with fixed absolute tolerance 1e-5.
Retain deviations rather than retune or select a run. Do not compare the
released probe to the distinct historical matched-pooling probe as an identity
check. Independently replay every prediction at batch size 17 versus 64 with
maximum absolute vector tolerance 2e-6, recompute cosine in PyTorch double,
and verify exhaustive-permutation identities and document-paired intervals.

## Inputs, execution, and artifact roles

The three candidate probes are downloaded from the existing `v1-paper` release
and independently verified by role, seed, byte size, SHA-256, TRAIN fit-group
hashes, and schedule hash. Existing final checkpoint bytes are unchanged.
The original official and both preceding additive namespaces remain untouched.
Zero merger/decoder/probe updates, teacher calls, classifier fits, or new merger
forward calls are required. Model evaluation uses FP32, no TF32, and inference
mode. Config/source identities are recorded before prediction generation.

```bash
.venv/bin/python -m pytest tests/test_reconstruction_calibration.py -q
.venv/bin/python scripts/evaluate_reconstruction_calibration.py all --device cuda
.venv/bin/python scripts/render_reconstruction_calibration.py
.venv/bin/python scripts/audit_reconstruction_calibration.py --device cuda
```

Outputs are create-only under `results/reconstruction_calibration_v1/`.
These commands record the run sequence, not permission to overwrite published
results. Exact input replay requires the local hashed caches and parent tensors
from improvements 1/2, plus released checkpoints/probes. Scalar reports alone
are not these tensors. An independent numerical replication must preserve the
published artifacts in a separate workspace and pin its own regenerated input
fingerprints before producing new predictions; it must not bypass integrity
checks or mix regenerated and historical scores.
