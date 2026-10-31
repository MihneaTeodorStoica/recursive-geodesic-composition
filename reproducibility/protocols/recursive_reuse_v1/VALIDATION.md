# Recursive-reuse validation record

## Completed experiment

- 606 exactly-four-child prefixes; corpus counts 148 / 133 / 142 / 183;
  all document-order hashes match the frozen earlier exact-K4 population.
- Three released final checkpoints, unchanged and independently hash-verified.
- Same cached leaves, targets, and token masses as improvement 1; no downloads,
  teacher encoding, merger/decoder updates, or classifier fits in this stage.
- Direct four-child composition and balanced binary reuse of the same K4
  checkpoint; two active leading slots at every binary node.
- Exact intermediate counts and document alignment preserved. All intermediate
  and final vectors are finite and unit-normalized.
- Direct and flat-pooling vectors are bitwise identical to the corresponding
  improvement-1 subset. Independent three-call tree replay on all documents
  and checkpoints, using batch size 17 rather than 64, differs by at most
  `8.940696716308594e-08`, below the fixed `2e-6` tolerance.
- Cosines independently recomputed in PyTorch double precision; paired
  differences, 2,000-replicate intervals, and degradation counts reproduced.
- No empirical-launch recovery, changed tolerance, outcome-based rerun,
  alternative-tree selection, or discarded document was needed.

## Main outcome (checkpoint scores averaged within each document)

| Quantity | Value |
| --- | ---: |
| Direct teacher cosine | 0.977575762883 |
| Recursive teacher cosine | 0.977114453581 |
| Recursive minus direct | -0.000461309302 |
| Paired 95% interval | [-0.000536902955, -0.000389935553] |
| Direct–recursive output cosine | 0.999441400752 |
| Recursive minus flat weighted pooling | +0.005612296188 |
| Paired 95% interval vs weighted pooling | [+0.005340518723, +0.005899458238] |
| Documents degraded | 454 / 606 (74.9%) |
| Documents losing more than 0.001 | 148 / 606 (24.4%) |
| Checkpoint-mean loss 95th percentile | 0.001938610587 |
| Maximum checkpoint-mean loss | 0.004846909627 |
| Largest individual-checkpoint loss | 0.008064908884 |
| Direct / recursive ms per document | 0.078448 / 0.205201 |
| Recursive / direct batched cost | 2.615749x |

Every checkpoint and every corpus has a negative mean recursive-minus-direct
contrast. This is a small but systematic loss on the tested tree, not exact
invariance. Both learned routes exceed both flat pooling baselines on this
subset. No inference is made about unrestricted-root teacher fidelity, unseen
trees/depths, or recursive source-detection AUC.

## Full-document accounting

All complete global token, leaf-count, and K4-depth histograms match each of the
three original official records. Per-corpus metadata is also cross-checked.

- Depth 1 / 2 / 3: **3,631 / 349 / 20** documents.
- Actual natural-K4 recursive reuse: **369 / 4,000 (9.225%)**.
- Strictly above documented 8,192-token context: **81 / 4,000 (2.025%)**.
- Length-bin counts: **2,707 / 924 / 288 / 61 / 15 / 5** for bins ending at
  2,048 / 4,096 / 8,192 / 16,384 / 32,768 / infinity.
- Total: 9,142,043 tokens; 11,356 leaves; 4,789 internal merges per checkpoint.
- These are descriptive full-document counts, not a new semantic full-arm run.

## Software and manuscript validation

```bash
.venv/bin/python -m pytest -q
# 93 passed
.venv/bin/python -m compileall -q src scripts tests
uv pip check --python .venv/bin/python
# 96 packages checked; all compatible
.venv/bin/python scripts/audit_recursive_reuse.py --device cuda
# passed, including independent replay of every document/checkpoint
artifacts/tools/tectonic --keep-logs --keep-intermediates paper/main.tex
git diff --check
```

The manuscript builds cleanly without TeX warnings into a **27-page PDF**.
The abstract, methods, new results/ECDF, population counts, and appendix table
were visually inspected in the rendered PDF. The source now distinguishes a
tree/active-arity change from a depth-only effect, and checkpoint-mean losses
from individual-checkpoint tails. Tests check table values and reported scores
against the full-precision artifacts. The original 71 tests continue to pass.

Original official result SHA-256 remains
`2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917`.
Improvement-1 result SHA-256 remains
`f16d23f504d3155dce680243a680d6c26ed2835b46f6aef9100b7fd9c1c57da5`.
New result SHA-256:
`74944d0960b9311ef8b897ef130474b6a04d463cac4ea6222cb8d96aa6614e29`.

The original official evaluation and improvement-1 result namespaces are
unchanged. Improvement 3 (reconstruction calibration) has **not** been started.
