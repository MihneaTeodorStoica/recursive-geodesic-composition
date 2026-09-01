# Publication validation

Validation was performed in a fresh `uv` virtual environment created with CPython 3.12.14 from `requirements.txt`.

## Environment

- Python: 3.12.14
- PyTorch: 2.13.0+cu130
- NumPy: 2.5.1
- CUDA visible: yes
- Direct dependency pins: those in `pyproject.toml`
- Dependency consistency: `uv pip check` passed for 96 installed packages

This matches the official result's recorded Python/PyTorch/NumPy versions. It does not reconstruct every transitive system component of the recorded environment.

## Completed checks

1. `python -m compileall -q src scripts tests figures` — passed.
2. Imports of the package and core training/evaluation modules — passed.
3. Editable installation plus source and wheel builds with `uv build` — passed.
4. `pytest -q` — **46 passed**.
5. All nine YAML configs and `CITATION.cff` parsed successfully; the CFF date is `2026-09-01` and its preferred-citation title matches the final paper exactly.
6. Local Markdown link validation — passed for 18 links.
7. `--help` import/argument smoke for the training, K-ary, official, final-v6, and release-materialization entry points — passed.
8. Exact construction of the final 768-dimensional, 1024-hidden, eight-layer, **16-head**, FFN-3072, K=4 strict-autoencoder model — passed.
9. Synthetic direct merges with two, three, and four active K4 slots, strict-decoder forward, and recursive reduction beyond four leaves — passed with finite unit-norm roots and conserved counts.
10. Frozen official result parsing and publication numerical/accounting checks — passed.
11. The release manifest contains exactly 12 binary artifacts: three final checkpoints, three continuation source checkpoints, and six fitted probes across seeds 424242, 271828, and 314159.
12. Every binary matched its manifest role, seed, byte size, and SHA-256. Probe payload fields confirmed candidate/source role, `fresh_v6_train` fit role, and seed; identity was not inferred from filenames.
13. The strict official preaccess verifier loaded all six model checkpoints and all six probes successfully, confirmed finite tensors and 19 strict-decoder state tensors per seed, and returned `preaccess_verified=true` with no official data/cache access.
14. The publication artifact checksum file verified all 12 release binaries.
15. Figure 2 and Figure 3 regenerated successfully from frozen machine-readable inputs with no result-value fallback.
16. Publication-copy hash verification with `sha256sum --check reproducibility/hashes.txt` — passed.
17. Final text, SVG, PNG metadata, binary-string, tracked-tree, and release-metadata scans for the prohibited name/link patterns — zero matches.
18. Git history validation — `main` has one root commit, annotated tag `v1-paper` targets it, and no `.pt` file is present in ordinary Git history.

## v1-paper release checkpoint smoke

`autoencoder_seed424242.pt` was downloaded from the final [`v1-paper` release](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v1-paper) and checked independently:

- bytes: `494519415`
- SHA-256: `989f9bc496bea484974679cd181e43836caf3283ffdc8e66507046945e0902de`
- checkpoint seed: `424242`
- checkpoint variant: `autoencoder`
- model: d=768, hidden=1024, layers=8, heads=16, FFN=3072, K=4
- decoder: strict parent-only, output shape `(1, 4, 768)`
- geometry: tangent expmap, maximum angle 0.5 rad
- synthetic direct root: shape `(1, 768)`, finite, norm `1.000000000`
- synthetic five-leaf recursive root: shape `(1, 768)`, finite, norm `1.000000000`, two reduction levels

The state dictionary loaded with `strict=True` through the publication model builder.

## Figure regeneration comparison

Only the final paper's two external figures are present; Figure 1 is generated in LaTeX/TikZ.

- Regenerated Figure 2 and Figure 3 PNG dimensions and RGBA pixels are exactly identical to the corresponding pre-publication plots (`0` changed channels; maximum absolute pixel difference `0`).
- PNG metadata now records author `Mihnea-Teodor Stoica` and creator `recursive-geodesic-composition/figures/generate_figures.py`.
- SVG metadata uses the same neutral creator and deterministic hash salt `recursive-geodesic-composition-v1-paper`.
- Current PNG/SVG hashes are recorded in `reproducibility/hashes.txt` and `reproducibility/provenance.md`.

## Normalized result validation

Four publication JSON copies had only an irrelevant absolute workspace prefix removed from initialization-checkpoint path fields. Structural comparison confirmed that no other field changed. Their original artifact hashes and new publication hashes are recorded in [`provenance.md`](provenance.md). Active official provenance checks use the publication hashes and passed with all exact model/probe hashes unchanged.

## Checks not run

The teacher model and datasets were not downloaded. The clean TRAIN cache and official cache were not regenerated, so continuation training and official numerical evaluation were not rerun. No official metric is claimed as newly recomputed.

Remaining practical requirements are deterministic cache regeneration from pinned inputs, dataset availability/streaming behavior, and compatible software/GPU execution. Official bootstrap intervals remain conditional on the three fixed checkpoints. The unrestricted full-document arm intentionally has no full-document teacher target or associated teacher cosine/AUC.
