# Improvement 1: completed validation

Branch: `revision/safe-arm-baselines`. Only the matched safe-arm baseline
improvement has been implemented; recursive-reuse and reconstruction experiments
have not been started.

## Execution

- Installed the pinned environment using Python 3.12.14; `uv pip check` passes
  for 96 packages. GPU: NVIDIA GeForce RTX 4090 Laptop GPU; PyTorch 2.13.0+cu130.
- Downloaded and independently SHA-256-verified all three final `v1-paper`
  checkpoints. No merger or decoder updates were performed.
- Regenerated the safe cache from the pinned upstream datasets/model without
  requiring original experiment files. All four 1,000-document order hashes,
  pooled order, prefix-token totals, and active-arity counts match the original
  publication. Original embedding byte identity is not claimed.
- Completed nine representation instances (three deterministic baselines plus
  two learned outputs for each of three checkpoints), 45 representation/scope
  evaluations, and 90 fixed classifier fits.
- Completed shared-document paired bootstraps, per-corpus/arity summaries,
  shuffled-teacher calibration, and same-protocol composition timing.

The initial preparation dtype preflight failed before producing safe embeddings
or comparison scores. Its correction is documented in
[`PREPARATION_RECOVERY.json`](PREPARATION_RECOVERY.json); the original loader
was retained unchanged, and no outcomes were available for selection.

## Checks

- `python -m pytest -q`: **71 passed**.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed.
- `scripts/audit_safe_baselines.py`: passed. This verifies document identity,
  independently recomputed cosines, paired differences and intervals, saved
  classifier scores, split integrity, aggregate AUCs, and artifact identities.
- All 55 entries in the revision result hash manifest verify locally, including
  the ignored vector binary and the published regenerated-cache manifest.
- Original official result SHA-256 remains
  `2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917`.
- Original merger/teacher code, official protocols/configs/results, and released
  checkpoint bytes are unchanged.

The audit's integrity pass does **not** mean exact numerical reproduction of the
old run: all pooled learned values meet the declared comparison tolerances, but
one of 15 checkpoint/scope AUC comparisons does not. FineWeb seed 271828 differs
by `-0.000576`, beyond the declared `0.0005` diagnostic tolerance. The maximum
absolute teacher-cosine difference across those scopes is approximately
`0.00000491446`. These differences are retained in the report and manuscript;
no tolerance was changed and no replacement run was selected.

## Paper

- Updated the abstract, evaluation protocol, results, discussion, limitations,
  future work, conclusion, and reproducibility appendix to reflect the measured
  baseline comparison rather than the old missing-baseline limitation.
- Main matched table is generated from machine-readable results; tests check its
  values and the manuscript's paired intervals against those results.
- Built the 24-page `paper/main.pdf` with Tectonic 0.17.0. No unresolved references,
  overfull boxes, or LaTeX warnings were reported. Rendered pages 12 and 21 were
  inspected for the main comparison/contrast tables and the per-seed appendix.
- Existing reconstruction, unrestricted-recursion, and original official values
  are preserved and distinguished from the additive regenerated-input study.

Results and caveats are in
[`../../../results/safe_baselines_v1/REPORT.md`](../../../results/safe_baselines_v1/REPORT.md).
Checkpoints, caches, vector tensors, logs, and PDF build products remain local
ignored artifacts, not Git objects.
