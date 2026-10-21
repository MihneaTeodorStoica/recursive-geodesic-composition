# Recursive Geodesic Composition

Publication repository for **One Vector from Many: Hierarchical Long-Document Representation through Recursive Geodesic Transformer Composition**, by **Mihnea-Teodor Stoica**.

This repository contains the final `clean-v6` K=4 strict-autoencoder implementation, the `official-v6` protocol and frozen results, publication figures, and additive reviewer-response controls. The scientific experiment lineage remains `clean-v6`/`official-v6`. `v3.3-paper` pins corrected documentation alongside the unchanged v3.2 manuscript formats and consolidated attachments from earlier releases. Older releases and their download links remain available. No manuscript revision replaces those historical results or artifacts.

## Paper

The [LaTeX manuscript](paper/main.tex), [bibliography](paper/references.bib), and bundled figures are in [`paper/`](paper/). See the [paper README](paper/README.md) for build instructions and import provenance. The [evidence/limitations map](paper/REVIEW_EVIDENCE_MAP.md) separates historical results, additive controls, downstream tasks, model-training interventions, and what remains unknown.

### Current release: v3.3-paper

Read the [arXiv-compatible named preprint](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/download/v3.3-paper/One_Vector_from_Many_arxiv.pdf), [anonymous TMLR submission PDF](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/download/v3.3-paper/One_Vector_from_Many_tmlr_submission.pdf), or [release notes](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v3.3-paper). This documentation-only release preserves the v3.2 PDFs byte-for-byte. The October 21 manuscript revision updates citations, the introduction roadmap, terminology, Figure 1 descriptions, and repeated prose. The named preprint includes a contact email and October 2026 date; the TMLR submission remains anonymous. Limitations retain visible *Proposed remedy* labels. No experiments, reported numerical results, or trained weights changed; the paper does not establish a consistent retrieval advantage over averaging.

```bash
gh release download v3.3-paper \
  --repo MihneaTeodorStoica/recursive-geodesic-composition \
  --dir artifacts/v3.3-paper
(cd artifacts/v3.3-paper && sha256sum --check PAPER_FORMAT_ASSETS.sha256)
```

The arXiv-compatible preprint is a 38-page named, single-column article; the 40-page submission uses the official anonymous TMLR style. The `v3.3-paper` release contains the same 75 assets as v3.2: the two current PDFs and their checksum, all 70 unchanged v1/v2 attachments, and the original v3 PDF and checksum. Historical checksum files retain their original scope. See [release documentation](paper/README.md#v33-paper-release), [PDF build provenance](paper/README.md#v32-paper-release), and the [historical publication-check caveat](paper/README.md#v3-paper-release). Neither format has been submitted by this release workflow.

## Final system

- Frozen teacher: `Alibaba-NLP/gte-modernbert-base`
- Teacher revision: `e7f32e3c00f91d699e8c43b53106206bcc72bb22`
- Embedding dimension: 768
- Hidden/model dimension: 1024
- Transformer layers: 8
- Attention heads: 16
- FFN dimension: 3072
- Maximum merger arity: K=4
- Active arities: 2, 3, and 4 contiguous children; trailing slots are masked
- Reduction: deterministic contiguous K4 groups with singleton carry
- Geometry: normalized weighted anchor plus bounded tangent exponential-map correction
- Maximum geodesic angle: 0.5 rad
- Decoder: strict parent-only ordered four-slot decoder
- Final seeds: `424242`, `271828`, `314159`
- Final training: continued autoencoder retraining with 300 decoder-only and 100 joint updates; it was **not** training from scratch

The authoritative official-v6 result is [`results/official_v6/results.json`](results/official_v6/results.json), preserved byte-for-byte. Its SHA-256 is `236081e0c721602c742b573e3dfd06a556f9448ddaab299f10007788de51a7c0`.

### Official-v6 summary

| Metric | Publication value |
| --- | ---: |
| Three-seed strongest orientation-safe pooled AUC | 0.584232 |
| Shared-held-out-document-group bootstrap 95% CI | [0.578232, 0.590338] |
| Three-seed mean pooled teacher cosine | 0.985045 |
| Per-seed pooled AUC (424242 / 271828 / 314159) | 0.582809 / 0.582125 / 0.587763 |
| Per-seed pooled cosine (same order) | 0.985049 / 0.985033 / 0.985052 |

For seed 271828, the frozen JSON stores `0.5821244999999999`; `0.582125` is the final paper's displayed rounding of that same result. The frozen report remains unchanged.

Per-corpus three-seed AUC means are FineWeb 0.572271, FineWeb-Edu 0.566317, OpenWebText 0.583503, and Wikipedia 0.636689. Each unrestricted full-document seed accounts for all 4,000 documents, 9,142,043 tokens, 11,356 leaves, and 4,789 internal merges, with maxima of 58,731 tokens, 58 leaves, and depth 3. These values are parsed from the frozen JSON, not recomputed metrics.

## Repository structure

```text
paper/                           manuscript, bibliography, bundled figures, build instructions
configs/                         final, official, and K-ary validation configs
figures/                         external final-paper Figures 2 and 3 plus generator
results/
  official_v6/                   frozen official safe/full result and report
  clean_v6_development/          development results and checkpoint provenance
  kary_scaling/                  validation-only K=2/K=4/K=8 study for Figure 3
scripts/
  final_v6/                      clean-v6 preparation/freezing/evaluation protocol
  official_v6/                   frozen official protocol, initial failure, and recovery
src/recursive_geodesic_composition/
                                 model implementation and dependency closure
tests/                           model, protocol, and publication smoke tests
reproducibility/                 provenance, release manifest, hashes, and validation
```

The model, decoder, spherical operations, and recursive reduction remain together in `architecture.py` because that is the architecture used by the experiments.

## Review-control release (v2-paper-review)

The additive frozen-model controls include 24-permutation input-order tests, all five ordered binary trees and resegmentation, three matched TRAIN-only probe families, a four-task pinned full-corpus LongEmbed retrieval study, and a *new*, separately trained attached-versus-detached decoder-parent gradient-path intervention. They **do not** recreate the historical continuation or provide a from-scratch lineage. In LongEmbed, recursive minus weighted-pooling nDCG@10 is +0.01250 / −0.01995 / −0.02700 / −0.00448 (2WikiMQA / QMSum / SummScreenFD / NarrativeQA); there is no across-task improvement. The documented CLS interface is evaluated with matching CLS queries and documents, not silently mixed with mean-pooled inputs. See [study setup and recoveries](reproducibility/protocols/review_completion_v1/README.md), [independent validation](reproducibility/protocols/review_completion_v1/VALIDATION.md), and [the separate new-artifact manifest](reproducibility/REVIEW_RELEASE_MANIFEST.json).

The new release carries the reviewed PDF and the **newly trained** model/probe weights; the historical `v1-paper` checkpoints and probes stay in their own release. New evaluation scores, provenance, audit checks, and immutable checksums are tracked as JSON in the repository. Large derived embedding vectors, training caches, and source-corpus text are **not** release assets: the results identify their SHA-256 identities, and the scripts regenerate them with pinned data/model revisions. This is not a claim of byte-identical historical embeddings. For downloaded review-release assets use `scripts/materialize_review_release.py --verify-only <asset-directory>` or `--mode hardlink` to place only missing weight files at their recorded paths. The release's `REVIEW_RELEASE_ASSETS.sha256` verifies the PDF and weights.

The original `v1-paper` models and probes remain required to regenerate the comparisons. On the reconstructed full environment, `python scripts/audit_review_completion.py` checks all five additional namespaces. To reproduce the historical manifest-hashed PDF, first check out `v2-paper-review`, then run `cd paper && SOURCE_DATE_EPOCH=1790294400 tectonic -X compile main.tex`. That recipe does not describe the current preprint; see the [v3 build instructions](paper/README.md#v3-paper-release).

## Setup

The recorded official environment used Python 3.12.14, PyTorch 2.13.0+cu130, NumPy 2.5.1, and CUDA.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

An equivalent `uv` setup is:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Teacher and dataset downloads require network access and acceptance of their hosting terms. No access token is stored in this repository.

## v1-paper release artifacts

The [`v1-paper` GitHub Release](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v1-paper) contains every required final-system checkpoint and fitted probe. `.pt` files are release assets, not ordinary Git objects. Roles and identities are declared in [`reproducibility/ARTIFACT_MANIFEST.json`](reproducibility/ARTIFACT_MANIFEST.json); checksums are in [`reproducibility/ARTIFACTS.sha256`](reproducibility/ARTIFACTS.sha256).

### Final checkpoints

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `autoencoder_seed424242.pt` | 494519415 | `989f9bc496bea484974679cd181e43836caf3283ffdc8e66507046945e0902de` |
| `autoencoder_seed271828.pt` | 494519479 | `a4cb1d79a7ae1bb0ca8855e3605c1ff50e761ca17ada39aa6b0237900dce7528` |
| `autoencoder_seed314159.pt` | 494519479 | `45a80bc3005bdf79fcafe3fce65b927a8f39e7ec2c019a34c3e9d1c8ce327a07` |

### Continuation source checkpoints

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `source_seed424242.pt` | 494517367 | `2365a789f61ba6dab9967041b8dcee2a19dc969fd97b71026bdee970f7ea4797` |
| `source_seed271828.pt` | 494517431 | `63103ce0b940dab6eec02c62ae5e177b0f23941f1775624b2fe3b27f60105603` |
| `source_seed314159.pt` | 494517431 | `27116bb5dbe279750264147100b38323ec432329b385404f59db5bf753d4b3a2` |

### Fitted parent-only probes

| Role | Seed | Asset | Bytes | SHA-256 |
| --- | ---: | --- | ---: | --- |
| candidate | 424242 | `candidate_parent_only_probe_seed424242.pt` | 7889085 | `a9e30ac9df31af63c525bdf08e812d5562bf2ba7c4c60ea8f439f4e4492e1943` |
| source | 424242 | `source_parent_only_probe_seed424242.pt` | 7889049 | `7d71e0951ebace7c65dfbae7ca54a23f9c972d63ba308eb7d4b9f727d9a6e830` |
| candidate | 271828 | `candidate_parent_only_probe_seed271828.pt` | 7889085 | `7eff7a553a7dc17d2f75127b6622067f2f11add3454884b0c532782428e98fb9` |
| source | 271828 | `source_parent_only_probe_seed271828.pt` | 7889049 | `b3f422a0cc7c8ff5cb9ce1e066f38b51f004417d6d0d74f4e974f78eeeadbc8e` |
| candidate | 314159 | `candidate_parent_only_probe_seed314159.pt` | 7889085 | `14736e8cbdc623cd004ef031f8af719dffa0e627c37425611e9ede6df2bd514a` |
| source | 314159 | `source_parent_only_probe_seed314159.pt` | 7889049 | `617cd5430ed9751121e3f222de942ea8f4e5f4164ce9a0514e36ae88bb954497` |

Probe role and seed were verified from the recovered manifest and checkpoint payloads, not inferred from filenames alone.

Download and verify the complete bundle with GitHub CLI:

```bash
mkdir -p artifacts/v1-paper
gh release download v1-paper \
  --repo MihneaTeodorStoica/recursive-geodesic-composition \
  --dir artifacts/v1-paper
(cd artifacts/v1-paper && sha256sum --check ARTIFACTS.sha256)
python scripts/materialize_release_artifacts.py \
  artifacts/v1-paper --verify-only
```

The frozen protocols expect their original relative artifact paths. To verify and place all assets there without duplicating bytes on the same filesystem:

```bash
python scripts/materialize_release_artifacts.py \
  artifacts/v1-paper --mode hardlink
```

Use `--mode copy` across filesystems. The materialized `.pt` paths remain ignored by Git.

## Inference from already-computed leaves

Download one exact final checkpoint:

```bash
mkdir -p checkpoints
gh release download v1-paper \
  --repo MihneaTeodorStoica/recursive-geodesic-composition \
  --pattern autoencoder_seed424242.pt \
  --dir checkpoints
printf '%s  %s\n' \
  989f9bc496bea484974679cd181e43836caf3283ffdc8e66507046945e0902de \
  checkpoints/autoencoder_seed424242.pt | sha256sum --check
```

Then use ordered, already-computed 768-dimensional leaf embeddings and exact positive token counts:

```python
from pathlib import Path

import torch

from recursive_geodesic_composition.architecture import recursive_reduce
from recursive_geodesic_composition.evaluate_kary_validation import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = build_model(
    Path("checkpoints/autoencoder_seed424242.pt"), device
)
leaves = torch.load("leaves.pt", map_location="cpu").to(device)        # [N, 768]
counts = torch.load("token_counts.pt", map_location="cpu").to(device) # [N]

with torch.inference_mode():
    root, levels = recursive_reduce(
        model,
        leaves.unsqueeze(0),
        counts.to(torch.float32).unsqueeze(0),
    )
    reconstructed_slots = model.decode(root)

assert root.shape == (1, 768)
assert reconstructed_slots.shape == (1, 4, 768)
assert torch.isfinite(root).all()
assert torch.allclose(root.norm(dim=-1), torch.ones(1, device=device), atol=2e-5, rtol=0)
```

Counts are token masses, not learned semantic weights. Leaves must remain ordered and contiguous. Active children occupy a contiguous prefix; inactive trailing counts are zero. Recursive reduction carries singleton remainders without a synthetic merge.

## Continuation training

The three clean-v6 models continue from the exact released source checkpoints. For each seed, source decoder keys are ignored, a strict decoder is deterministically initialized, the decoder trains alone for 300 updates, and encoder plus decoder train jointly for 100 updates with encoder learning-rate scale `0.02`. The fixed step-400 checkpoint is frozen before development data is prepared. Exact per-seed configs are in [`configs/final_v6/`](configs/final_v6/).

The authoritative configs allocate 80 disjoint TRAIN groups per corpus for probe fitting and 80 for floors. Earlier 120+120 development configs were superseded before model training because FineWeb contained only 184 eligible exact-four groups.

The source checkpoints are available; the pinned TRAIN cache is deliberately not distributed and must be regenerated from the pinned inputs. In a clean working tree, materialize only source checkpoints before training:

```bash
python scripts/materialize_release_artifacts.py artifacts/v1-paper \
  --role continuation_initialization_source_checkpoint --mode hardlink

python scripts/final_v6/prepare_train.py \
  --config configs/final_v6/seed424242.yaml --device cuda

for seed in 424242 271828 314159; do
  python scripts/train_final_v6.py \
    --config "configs/final_v6/seed${seed}.yaml" \
    --variant autoencoder --seed "$seed" --device cuda
done

python scripts/final_v6/freeze_training_and_probes.py --device cuda
```

The deterministic clean TRAIN cache had 363,278,219 bytes in the recorded run. It is not a release asset. Continuing through development additionally requires regenerating the fresh development cache with the frozen scripts. Random initialization would be a different experiment.

## Official safe and unrestricted full-document evaluation

Official-v6 used the same 4,000 accepted documents in two mandatory arms:

1. **Safe teacher-comparable arm:** one direct K4 merge over an exact 1,025–4,096-token prefix, with a direct frozen-teacher target and fixed grouped logistic/RBF attacks.
2. **Unrestricted full-document arm:** every available token represented by exact contiguous leaves of at most 1,024 tokens, with no token, leaf, or depth cap, no truncation, and no full-root teacher target or AUC.

The initial evaluator created its one-shot marker and then failed before loading a safe tensor or producing a metric because two stale provenance lookups remained. The frozen recovery changed only those lookups and reused the prepared cache without a source rescan. Both evaluator versions and the recovery declaration are retained.

All exact final checkpoints, source checkpoints, and six fitted probes are available. The pinned official cache is deliberately not distributed and must be regenerated:

```bash
python scripts/materialize_release_artifacts.py artifacts/v1-paper --mode hardlink
python scripts/official_v6/verify_preaccess.py
python scripts/prepare_official_v6.py \
  --config configs/official_v6.yaml --device cuda

# Recorded expected failure before any metric is produced.
python scripts/official_v6/evaluate_official_test_v6_initial.py \
  --config configs/official_v6.yaml --device cuda

# Frozen evaluator-only recovery; evaluates both mandatory arms.
python scripts/evaluate_official_v6.py \
  --config configs/official_v6.yaml --device cuda
```

The recorded official cache had 77,225,389 bytes. It is not a release asset. The cache must be regenerated from the pinned corpus revisions and verified by the frozen protocol. This publication cleanup did **not** regenerate that cache or rerun the official numerical evaluation.

## Publication figure artifacts

The original publication snapshot records these figure artifacts:

1. Figure 1: architecture diagram generated directly in LaTeX/TikZ in the paper source;
2. Figure 2: [`fig2_official_per_corpus.png`](figures/fig2_official_per_corpus.png) / [`SVG`](figures/fig2_official_per_corpus.svg);
3. Figure 3: [`fig3_kary_scaling.png`](figures/fig3_kary_scaling.png) / [`SVG`](figures/fig3_kary_scaling.svg).

The imported [manuscript](paper/main.tex) renders its current figures inline with TikZ/PGFPlots. Its bundled PNGs, including `fig4_k4_pooling.png`, are preserved separately in [`paper/figures/`](paper/figures/) and are not referenced by the current TeX source. They do not replace the frozen assets above. Regenerate the two external figures in the top-level `figures/` directory from frozen machine-readable inputs with:

```bash
python figures/generate_figures.py
```

The generator has no hard-coded result fallback. Publication metadata identifies Mihnea-Teodor Stoica and this repository's generator. Figure hashes are recorded in [`reproducibility/hashes.txt`](reproducibility/hashes.txt).

## Pinned datasets

All final train/development/official configs use equal corpus weights and these revisions:

| Corpus | Hugging Face dataset/config | Revision |
| --- | --- | --- |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` / `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| FineWeb | `HuggingFaceFW/fineweb` / `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` |
| OpenWebText | `Skylion007/openwebtext` | `79d93d786212f7344586290adb811d4ae6a1762c` |
| Wikipedia | `wikimedia/wikipedia` / `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |

Clean-v6 used TRAIN `[140000,152000)`, development `[152000,161000)`, untouched buffer `[161000,162000)`, and official scan `[162000,192000)` in source-index space for each corpus.

## Reproducibility status

| Task | Exact released artifacts | Remaining requirement |
| --- | --- | --- |
| Final inference | three final checkpoints | download a checkpoint; provide/regenerate ordered teacher leaves and token counts |
| Continuation training | three source checkpoints | regenerate the pinned 363,278,219-byte TRAIN cache and run in the pinned environment |
| Official-v6 evaluation | three final checkpoints and all six fitted probes; source checkpoints are also available for strict preaccess checks | regenerate the pinned 77,225,389-byte official cache |
| Missing required binary artifacts | **none** | — |

The release does not include caches, raw datasets, full-document shards, TensorBoard events, giant intermediate tensors, or logs. Dataset streaming availability/order, hosting behavior, software/GPU differences, and deterministic cache regeneration remain practical constraints. Official bootstrap intervals are conditional on the three fixed checkpoints and do not measure retraining uncertainty. The unrestricted arm deliberately has no full-document teacher target, full-root teacher cosine, or generated-versus-teacher AUC.

Figure-content regeneration and frozen-result inspection require no model download. A numerical rerun of the separate validation-only K-ary scaling study would require recreating its own checkpoints and matched cache; Figure 3 itself is reproducible from the included frozen report.

See [`reproducibility/provenance.md`](reproducibility/provenance.md), [`reproducibility/source_mapping.md`](reproducibility/source_mapping.md), and [`reproducibility/validation.md`](reproducibility/validation.md) for provenance and validation boundaries.

## Tests

```bash
pytest
sha256sum --check reproducibility/hashes.txt
```

The suite covers spherical geometry, strict decoder shapes, active arities 2/3/4, recursive reduction, count conservation, official protocol restrictions, final config invariants, exact 16-head model construction, synthetic forward execution, and publication artifact hashes.

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). No DOI, arXiv identifier, journal venue, or ORCID is asserted here. The repository carries an all-rights-reserved [`LICENSE`](LICENSE).
