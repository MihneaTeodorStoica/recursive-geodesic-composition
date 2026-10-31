# Reconstruction-calibration validation record

## Completed scope

- Same 606 exact-four safe prefixes and identical direct parents/children from
  improvements 1 and 2, checked by document identity and tensor equality.
- All three released final checkpoints and their corresponding released
  candidate TRAIN-fitted probes. The three small probe assets were downloaded
  from `v1-paper` and independently checked by size, SHA-256, role, seed,
  TRAIN-selection hashes, and schedule hash.
- Parent-copy, built-in strict decoder, frozen probe, weighted pooling copy,
  and unweighted pooling copy. No new merger forward calls, teacher encoding,
  classifier fits, or merger/decoder/probe updates.
- Exhaustive 24-permutation and nine-derangement target-slot scoring, keeping
  all model inputs and predictions fixed. Also report the oracle best assignment.
- Two primary same-parent gains, four pooling-copy contrasts, all per-corpus
  and per-checkpoint results, document-paired intervals, and empirical CDFs.
- Configuration and source hashes frozen before new predictions. No outcome-based
  reruns, alternative probe selection, discarded documents, or tolerance changes.

## Findings (checkpoint scores averaged within documents)

| Rule | Ordered child cosine | Gain vs learned-parent copy | Ordered minus all-24 shuffled |
| --- | ---: | ---: | ---: |
| Learned parent copy | 0.914021511 | 0 | 0 |
| Built-in decoder | 0.915064015 | +0.001042504 | +0.000000686 |
| Frozen candidate probe | 0.908105018 | -0.005916492 | +0.000908435 |
| Weighted pooling copy | 0.920365100 | +0.006343590 | 0 |
| Unweighted pooling copy | 0.922575508 | +0.008553997 | 0 |

- Built-in minus same-parent copy: **+0.001042504**, paired 95% CI
  **[+0.001025339, +0.001060005]**; positive on all 606 documents after averaging
  checkpoint scores. It remains below both untrained pooling-copy rules on
  every document under that aggregation.
- Frozen probe minus same-parent copy: **-0.005916492**, CI
  **[-0.006150681, -0.005693017]**; improves only **11/606** documents.
- Built-in ordered-minus-shuffled gap: **6.85549988e-7**, CI
  **[5.96446083e-7, 7.72482561e-7]**. This is a tiny positive gap, not exact
  permutation invariance or evidence of robust ordered-child recovery.
- Probe order gap: **0.000908435**, CI **[0.000699193, 0.001115978]**, despite
  its lower overall cosine. Even its oracle best-assignment mean (0.911371109)
  is below learned-parent copy (0.914021511).
- About **98.98%** of target directions are outside the residual angular cap
  around their respective parents. The loose oracle cap ceiling is 0.982790593;
  this is not an achieved score or an explanation of the observed deficit.
- A positive order gap can reflect positional priors rather than stored,
  document-specific order. These are local reconstructions, not recovery of
  all leaves from unrestricted roots.

The separately fitted historical matched-pooling probe binaries are absent
from the release. Their historical results remain unchanged and separate;
new pooling-probe gains or order scores are **not** inferred by mixing them
with regenerated copy results.

## Integrity, numerical replay, and historical discrepancies

Independent replay uses the strict-decoder component directly and batch size
17 instead of 64. Every prediction is replayed; maximum absolute difference
is **5.960464477539063e-8**, below the fixed 2e-6 tolerance. All models stay
within the residual angular bound. Actual maximum movements are about 1.10
and 10.83 degrees for the built-in and probe families, respectively.

Cross-slot cosines are recomputed independently in PyTorch double precision.
An independent permutation loop verifies all assignment means and the
all-24/off-diagonal identities. Copy baselines have zero order gap; unweighted
copy attains the shared-vector optimum. All paired scores and bootstrap
intervals are reproduced without fitting.

All six pooled means meet the fixed **1e-5** historical reproduction tolerance;
maximum pooled discrepancy is **4.140732e-6**. Six of 30 checkpoint/scope/method
comparisons exceed it: **OpenWebText, both decoder families, all three seeds**,
with positive differences from **1.156283e-5 to 1.304064e-5**. These remain
explicitly flagged in `RESULTS.json` and `REPORT.md`. The tolerance was not
changed, and the run was not selected or repeated to remove them. Original
embedding byte identity is unavailable, so their numerical cause is not
isolated. All new contrasts nevertheless use identical matched inputs.

## Software and manuscript

```bash
.venv/bin/python -m pytest -q
# 112 passed (including all 93 preceding tests)
.venv/bin/python -m compileall -q src scripts tests
uv pip check --python .venv/bin/python
# All 96 installed packages compatible
.venv/bin/python scripts/audit_reconstruction_calibration.py --device cuda
# passed
artifacts/tools/tectonic --keep-logs --keep-intermediates paper/main.tex
git diff --check
```

The manuscript builds without TeX warnings to a **29-page PDF**. The abstract,
methods, original-versus-new reconstruction tables, ECDF, and appendix were
visually inspected. Interpretation is updated in the introduction, results,
discussion, limitations, future work, conclusion, and analytical-bound appendix.

Original official result SHA-256:
`2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917`.
Improvement-1 result SHA-256:
`f16d23f504d3155dce680243a680d6c26ed2835b46f6aef9100b7fd9c1c57da5`.
Improvement-2 result SHA-256:
`74944d0960b9311ef8b897ef130474b6a04d463cac4ea6222cb8d96aa6614e29`.
New result SHA-256:
`6f378128f85bcac3ae762c9a931985e3fd5a110d08cc8334bcfef87746bc46c7`.

All preceding result namespaces and checkpoint bytes remain unchanged.
No subsequent improvement has been started.
