# Review source — One Vector from Many

**Hierarchical Long-Document Representation through Recursive Geodesic Transformer Composition**

This self-contained review copy contains the existing scientific implementation, exact experiment settings, frozen results and supporting audits for the separately submitted manuscript. No scientific source, reported numeric result, model weight or manuscript was changed for this rebuild. Identity/provenance redactions from the earlier package are documented separately.

## Quick start: no GPU, model download or project installation

From the review checkout root, or the extracted `TMLR_Supplement/` directory, with Python 3.12 or later:

```bash
python supplement.py verify
python supplement.py report
python supplement.py materialize weights \
  --role candidate_parent_only_fitted_probe \
  --role source_parent_only_fitted_probe --verify-only
```

These standard-library commands verify packaged hashes, replay saved scalar-score means and 120 classifier AUC point estimates, and verify the six included probe files. They **do not run model inference**, regenerate a corpus cache, fit classifiers or replay bootstrap intervals. `SHA256SUMS` additionally supports `sha256sum --check --quiet SHA256SUMS`. These checksums detect file changes; they are not authenticity signatures.

## Contents and manuscript correspondence

| Location | Contents |
| --- | --- |
| `src/recursive_geodesic_composition/` | All 19 original scientific modules: spherical/geodesic merger, recursive aggregation, strict parent-only decoder, losses, training, baselines and evaluations; byte-identical to the original source and prior supplement. |
| `configs/`, `scripts/` | Clean-v6 per-seed continuation, official-v6 evaluation, fixed attacks/bootstrap settings, data/cache preparation, scaling and additional controls. Metadata-only packaging changes are disclosed. |
| `results/` | All 12 tracked result namespaces, including aggregate JSON, saved document/query scalar scores, classifier predictions, TeX tables, plots, protocols and historical audits. |
| `weights/` | Six exact historical candidate/source fitted parent-only probes, 47,334,402 bytes total; **not** full model checkpoints. |
| `reproducibility/` | Exact weight manifests, historical protocol/recovery records, source mappings, package checksums and independent audit evidence. |
| `figures/`, `tests/` | Supporting frozen-result plots/generator and existing behavioral/protocol tests. No broad test-suite success is claimed for this rebuild. |

[EVIDENCE_MAP.md](EVIDENCE_MAP.md) maps **Tables 4–17**, manuscript Figures 1–4 and order, tree/segmentation, decoder-family and gradient-path controls to verified paths. The current manuscript's Figures 2–4 are the plots under `results/`; top-level `figures/` contains separate supporting official/scaling plots. The manuscript remains authoritative and is not duplicated or rebuilt here.

[REPRODUCING.md](REPRODUCING.md) contains inspected preparation, loading, inference, rendering, audit and continuation recipes. [VALIDATION.md](VALIDATION.md) distinguishes executed checks from retained historical verdicts. [ANONYMIZATION.md](ANONYMIZATION.md) documents transformations and residual linkability.

## Install the experimental environment

The recorded environment is Python 3.12.14, PyTorch 2.13.0+cu130, NumPy 2.5.1 and CUDA. Runtime versions are pinned in `pyproject.toml`; transitive resolutions are in `uv.lock`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full project requires Python `>=3.12,<3.13`; the standalone quick-start helper works independently of that installation. A compatible CUDA driver/runtime and substantial GPU resources are required for teacher encoding, model inference and training. Packaging/statistical checks use a separate host environment; the historical Torch environment was not recreated in this rebuild.

## Frozen teacher and datasets

Preparation scripts download the exact configured teacher/tokenizer and pinned upstream datasets. Network access, hosting availability and applicable terms remain prerequisites; no credentials or source text are included.

| Input | Identifier/configuration | Pinned revision |
| --- | --- | --- |
| Teacher | `Alibaba-NLP/gte-modernbert-base` | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| FineWeb | `HuggingFaceFW/fineweb`, `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` |
| OpenWebText | `Skylion007/openwebtext` | `79d93d786212f7344586290adb811d4ae6a1762c` |
| Wikipedia | `wikimedia/wikipedia`, `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |
| Retrieval control | `dwzhu/LongEmbed` | `10039a580487dacecf79db69166e17ace3ede392` |

Use the configuration-specific commands in REPRODUCING.md, not unpinned downloads or a class's legacy defaults. The main representation uses normalized masked-mean teacher embeddings; matched CLS references in the retrieval study are distinct interfaces. TRAIN/development/buffer/official source-index windows are `[140000,152000)`, `[152000,161000)`, `[161000,162000)`, and `[162000,192000)` respectively.

## Reproduction levels and missing weights

1. **Saved evidence.** Inspect retained JSON/TeX/plots and run the quick-start commands. Saved scalar-score replay does not independently recover omitted embeddings. Original official result files contain aggregates, not the prediction arrays needed for complete original bootstrap replay. Independent checks of saved baseline bootstrap intervals are recorded in VALIDATION.md.
2. **Inference and full-vector audits.** Obtain exact full models, regenerate pinned caches and run the appropriate evaluator. Frozen downstream configs can require historical upstream file identities; rerunning an upstream stage changes provenance bytes, so the supplied entrypoints are not a turnkey from-scratch chain. Work in a disposable copy and preserve the archived evidence and guards.
3. **Training.** The documented clean-v6 stage is a **300 decoder-only + 100 joint-update continuation from exact source checkpoints**, not training from random initialization. Complete historical source-checkpoint training lineage is not established. Newly trained probe/gradient controls are separate experiments.

`reproducibility/WEIGHTS_MANIFEST.json` identifies all 66 released `.pt` files by role, filename, size, SHA-256 and expected path. Six probes are included; the other 60 total **8,062,738,309 bytes** and exceed the [100 MB supplementary limit](https://jmlr.org/tmlr/author-guide.html). Final inference, source continuation and intervention weights are never substituted. The separately recorded scaling and matched-pooling artifacts have their own availability limitations.

**No verified anonymous weight endpoint is configured.** The source implementation and saved evidence are provided directly in this review snapshot; they do not depend on a weight-hosting service. An earlier full-repository mirror failed an anonymity/access audit and is not linked here; that historical failure is not a verdict on a newly verified mirror of this sanitized snapshot. For weight-dependent review, request the exact manifest-listed files through the anonymous submission discussion; [CHECKPOINT_HOSTING_PLAN.md](CHECKPOINT_HOSTING_PLAN.md) prioritizes useful collections and an author-mediated hosting plan. Full model-backed reproduction remains blocked until the required files are supplied.

For exact weights received in a flat local directory:

```bash
python supplement.py materialize /path/to/weights --role final_clean_v6_checkpoint --verify-only
python supplement.py materialize /path/to/weights --role final_clean_v6_checkpoint --mode hardlink
```

Use repeated `--role` for other selections and `--mode copy` across filesystems. The helper verifies bytes before placement at protocol/lookup paths and refuses different existing files. The included probes can be placed using the quick-start role selectors without `--verify-only`. The existing `build_model` loading example is in REPRODUCING.md. Load only trusted, verified checkpoint files.

## Limitations and anonymity

- Full-document execution has no direct full-root teacher target, teacher cosine or source-detection AUC. Prefix fidelity is not long-document semantic validation.
- Bootstrap intervals condition on fixed checkpoints/classifiers/splits, not complete source-lineage retraining uncertainty. LongEmbed shows no consistent across-task advantage over pooling.
- FP16 teacher execution, FP32 accumulation, batching, software/GPU differences and dataset streaming can change regenerated vectors. Seed 271828's AUC `0.5821244999999999` is displayed as `0.582125` in the manuscript.
- Historical source-hash discrepancies are investigated individually in [HASH_DISCREPANCY_AUDIT.md](HASH_DISCREPANCY_AUDIT.md). Preserving or recovering old source does not make omitted tensors available, and historical audit verdicts are not newly executed audits.
- Three omitted warm checkpoints contain personal paths in serialized metadata. Their exact original identities are retained with warnings; no sanitized replacement is represented as the original.
- Git history/remotes from the original source, identifying citation metadata, named PDFs/manuscript sources, personal paths and identifying figure metadata are excluded or sanitized in this review copy. Raw identity-bearing audit evidence and publication reports remain outside the branch. Public titles, scientific identifiers and exact artifact hashes remain searchable; no claim of perfect anonymity is made.

The main branch, existing release tags, submitted manuscript, experimental settings, reported results and original weights are unchanged. This review source is an independently rooted snapshot. The original all-rights-reserved software terms are retained with the copyright holder anonymized; publication or hosting does not change those terms.
