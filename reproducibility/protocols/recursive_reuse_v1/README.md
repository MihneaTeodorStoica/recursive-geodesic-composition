# Improvement 2: teacher-valid tree / recursive-reuse stress test

This additive post-review experiment reuses the verified cache and released
checkpoints from improvement 1. It does not alter either preceding result set,
fit classifiers, encode new teacher vectors, or update model parameters.

Completed: [report](../../../results/recursive_reuse_v1/REPORT.md),
[audit](../../../results/recursive_reuse_v1/AUDIT.json), and
[validation record](VALIDATION.md). The fixed design below was established
before generating the new recursive outputs.

## Fixed question and comparison

On the 606 previously identified exact-four-child safe prefixes, compare:

- `direct = f4(x1, x2, x3, x4)`;
- `recursive = f2(f2(x1, x2), f2(x3, x4))`.

Every call uses the **same frozen K4 checkpoint**. `f2` means two active leading
slots and two zero-mass padded slots; it is not a separately trained K2 model.
The first-level pairs are contiguous and ordered. Their counts are
`n1+n2` and `n3+n4`, and the root represents their sum exactly. The first level
batches both pairs together (2B nodes for B documents), followed by B root nodes.
There are three merge nodes per document rather than one.

This changes tree structure, active arity, generated-versus-leaf inputs, and
merge count together. Report it as a **tree/recursive-reuse stress test**, not a
controlled estimate of depth alone, associativity, or an intrinsic arity effect.
No alternative trees will be selected based on these outcomes.

Compare both roots against the identical cached teacher target and each other.
Include flat unweighted and token-weighted pooling computed from all four
original leaves, with one final normalization. Recursively normalizing pooled
intermediates while retaining only token counts is not equivalent to flat
pooling and must not replace that baseline.

## Population and metrics

The fixed corpus counts are FineWeb-Edu 148, FineWeb 133, OpenWebText 142, and
Wikipedia 183. Verify each exact-four document-order hash against the existing
matched-reconstruction protocol. Inputs are the same 3,073–4,096-token prefixes,
not unrestricted document roots—even when the source document is much longer.

Evaluate all three final checkpoints. Primary contrast:

`delta = cosine(recursive, teacher) - cosine(direct, teacher)`.

Negative delta indicates degradation. Also report:

- each root's teacher cosine and the direct–recursive output cosine;
- paired differences of both learned roots against both flat pooling rules;
- per-checkpoint and per-corpus results, with the same document resamples;
- per-document loss `-delta`, its min/5th/50th/95th/max percentiles, fractions
  degraded/improved, and losses exceeding 0.001, 0.005, and 0.01;
- an empirical loss CDF showing all documents, each checkpoint, and the
  checkpoint-mean scores;
- matched batched composition timing with device-resident original inputs,
  three warm-up and 20 measured passes, FP32, TF32 disabled, and batch size 64
  **documents**. The recursive first-level node batch is consequently 128.

Use 2,000 paired document bootstrap replicates with base seed 22092611 and fixed
scope offsets. Average checkpoint scores within each document before computing
the aggregate and its interval. There are 606 document units, not 1,818
independent observations; intervals condition on these checkpoints.

## Full-document population clarification

Use the existing 4,000 documents' full token counts for descriptive accounting,
without constructing new full-root teacher targets or running a new full arm.
Derive exact contiguous 1,024-token leaf counts, K4 maximum reduction depth,
and nonsingleton merge counts. Verify complete global token/leaf/depth
histograms against every seed in the frozen official result, and per-corpus
counts against the frozen summaries.

Report length bins 1,025–2,048, 2,049–4,096, 4,097–8,192, 8,193–16,384,
16,385–32,768, and >32,768; counts by depth; and the count **strictly above**
the backbone's documented 8,192-token context. Counts exclude special tokens.
The documented context threshold is not a guarantee that every shorter input
is numerically safe for this custom teacher procedure, nor the 4,096-token
experimental teacher-target cap. These are natural K4 full-document depths,
not the artificially imposed binary depth of the prefix stress test.

## Execution

Configuration and implementation are frozen into the new `ANALYSIS_PROTOCOL.json`
before generating the new outputs. The existing first-improvement working
artifacts are prerequisites: `cache_safe_baselines_v1/`, the three final
checkpoints, and the local `results/safe_baselines_v1/vectors.pt`. Their hashes,
document identities, and target alignment are verified; this stage performs no
network downloads. The Git-only scalar reports are not the input tensors.
Exact replay requires the archived first-improvement working artifacts; their
binary files are local, hash-recorded artifacts, not `v1-paper` release assets.
Independent regeneration from pinned upstream inputs is numerical replication,
not a promise of byte-identical cache files or manifests. Such replication must
use a separate workspace, preserve the published results, and explicitly pin
its own preceding-analysis fingerprints in the new protocol before inference;
do not bypass integrity checks or mix its embeddings with published scores.
The commands below record the completed run's sequence, not an instruction to
overwrite an existing result namespace.

```bash
.venv/bin/python -m pytest tests/test_recursive_reuse.py -q
.venv/bin/python scripts/evaluate_recursive_reuse.py all --device cuda
.venv/bin/python scripts/render_recursive_reuse.py
.venv/bin/python scripts/audit_recursive_reuse.py --device cuda
```

The output namespace is create-only. A completed inference can be followed by
`analyse` without recomputing outputs; the renderer and auditor do not select
or retune a tree. All outcomes are reported. The independent audit reconstructs
the tree with separate left/right calls and a different document batch size,
checks every document/checkpoint, independently recomputes cosine in double
precision, and recomputes paired intervals. A fixed maximum absolute vector
difference tolerance of `2e-6` applies to replay and to the prior direct/pooling
vectors. Checkpoint and preceding-result hashes are checked unchanged.
