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

The full project requires Python `>=3.12,<3.13`; the standalone quick-start helper works independently of that installation. Teacher encoding and training require compatible runtime/GPU resources. Later CPU verification safely loaded all 66 published checkpoints and exercised their model paths; the historical CUDA environment was not reproduced.

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

## Reproduction levels and public checkpoints

1. **Saved evidence.** Inspect retained JSON/TeX/plots and run the offline quick-start commands. These replay scalar means and 120 classifier AUCs, not fresh neural inference or original official bootstrap predictions.
2. **Model-backed work.** Download the required checkpoint groups and the separately pinned six recovered numeric caches described in CHECKPOINT_ACCESS.md, then follow REPRODUCING.md. Some study-specific inputs remain absent. Frozen downstream provenance guards and create-only namespaces remain; the official cache's anonymous metadata derivative does not satisfy original-byte guards.
3. **Training.** Clean-v6 is a **300 decoder-only + 100 joint-update continuation from exact source checkpoints**, not random initialization. Fifteen recovered primary records now trace the earlier sources through two encoder stages; earlier tensors and complete historical trainer/execution bindings remain missing. See `reproducibility/RECOVERED_SOURCE_LINEAGE.json`.

**Public weights:** [Hugging Face checkpoint repository](https://huggingface.co/tmlr-embedding-artifacts/recursive-geodesic-checkpoints)

**Pinned Hugging Face revision:** `9187157fa4fc6f790be1775475952f14ae33d02e`

**Anonymous source:** [Anonymous review repository](https://anonymous.4open.science/r/recursive-geodesic-composition-0F32/)

All **66 released artifacts** are hosted: **63 unchanged originals and three explicitly mapped metadata-only warm-state derivatives**. Their hosted sizes total **8,110,072,711 bytes**. Six original historical probes remain in this ZIP; the 60 external artifacts total **8,062,738,309 bytes** and are not added to the [under-100-MB supplement](https://jmlr.org/tmlr/author-guide.html).

```bash
# Standard library only; no account or token; verifies hashes and evaluation paths.
python scripts/download_checkpoints.py --group final --verify
python scripts/download_checkpoints.py --group source --group historical_probes --verify
```

[CHECKPOINT_ACCESS.md](CHECKPOINT_ACCESS.md) documents all groups, manual downloads, exact installation paths and preservation evidence. `reproducibility/HOSTED_CHECKPOINTS.json` pins the repository/revision and hosted sizes/hashes; `WEIGHTS_MANIFEST.json`, original release manifests and frozen experiment records retain the original artifact identities. Final, source and intervention roles are never substituted.

**Recovered caches:** six separately pinned bundles contain 8,032 unchanged tensor files, totaling 911,050,856 compressed bytes. `reproducibility/HOSTED_CACHES.json` pins revision `9e4c8e02375f66b3cf9918a20eb81446f996620e`, every extracted-file hash and two explicit official-cache metadata derivatives. No multi-gigabyte weights or caches are added to the supplementary ZIP. CHECKPOINT_ACCESS.md gives download/extraction commands, source attribution and guard limitations.

Nine separately documented older scaling/matched-pooling payloads remain unavailable. Raw corpus/task inputs and some other study-specific token/vector caches remain external. Sanitized warm or official-cache metadata derivatives cannot satisfy their original-byte provenance guards. Never rewrite an old experiment hash or bypass a guard to make a restored-artifact run appear to pass.

## Limitations and anonymity

- Full-document execution has no direct full-root teacher target, teacher cosine or source-detection AUC. Prefix fidelity is not long-document semantic validation.
- Bootstrap intervals condition on fixed checkpoints/classifiers/splits, not complete source-lineage retraining uncertainty. LongEmbed shows no consistent across-task advantage over pooling.
- FP16 teacher execution, FP32 accumulation, batching, software/GPU differences and dataset streaming can change regenerated vectors. Seed 271828's AUC `0.5821244999999999` is displayed as `0.582125` in the manuscript.
- Historical source-hash discrepancies are investigated individually in [HASH_DISCREPANCY_AUDIT.md](HASH_DISCREPANCY_AUDIT.md). Preserving or recovering old source does not make omitted tensors available, and historical audit verdicts are not newly executed audits.
- The three original warm checkpoints contain personal location metadata and are not publicly redistributed unchanged. Separately hashed, length-preserving metadata-only derivatives retain all tensor/optimizer state; original-byte provenance guards still reject those derivatives. See CHECKPOINT_ACCESS.md.
- Git history/remotes from the original source, identifying citation metadata, named PDFs/manuscript sources, personal paths and identifying figure metadata are excluded or sanitized in this review copy. Raw identity-bearing audit evidence and publication reports remain outside the branch. Public titles, scientific identifiers and exact artifact hashes remain searchable; no claim of perfect anonymity is made.

The main branch, existing release tags, submitted manuscript, experimental settings, reported results and original weights are unchanged. This review source is an independently rooted snapshot. The original all-rights-reserved software terms are retained with the copyright holder anonymized; publication or hosting does not change those terms.
