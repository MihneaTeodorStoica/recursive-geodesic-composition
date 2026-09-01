# Provenance

## Frozen experiment lineage

- Frozen experiment record commit: `32281e4a6836524448a61d87c5b4ed7eff32c0d3`
- Original experiment implementation path: `5/`
- Official result commit: `73f9da3`
- Official result tag: `official-k4-strict-ae-v6-result`
- Paper-preparation experiment tag: `recursive-geodesic-transformer-strict-ae-v6-paper`

### Final-v6 chain

1. Commit `f38f75d2d70a9af92d3bdeae959281173c7998ee` completed clean-v6 training/development. It froze three fixed step-400 checkpoints and TRAIN-only fitted-probe floors before development data was opened.
2. Commit `b499e93` and annotated tag `official-k4-strict-ae-v6-preaccess` froze the official protocol before scanning `[162000,192000)` for each corpus.
3. The first evaluator attempt failed with `KeyError: 'confirmation'` before loading a safe tensor or producing a metric. Commit `339c8a0` and tag `official-k4-strict-ae-v6-evaluator-recovery` froze a lookup-only recovery.
4. The recovery reused the already-prepared cache without a source rescan. Commit `73f9da3` recorded `results.json` and `REPORT.md`.
5. Commit `32281e4` generated the pre-publication figure artifacts from frozen machine-readable results.

Model, data, and training source blobs were unchanged from the clean-v6 commit through the frozen experiment record. The publication snapshot is the annotated `v1-paper` tag.

## Released exact binary artifacts

The [`v1-paper` release](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v1-paper) contains:

- three exact final clean-v6 checkpoints;
- three exact continuation source checkpoints;
- three candidate and three source fitted parent-only probes.

Every binary was matched to a recovered manifest entry by role, seed, byte size, and SHA-256 before publication. Probe identity was not inferred from filenames. The recovered manifest SHA-256 is `de4451cc57489619af4a4d788c71639de81968b42571280c3f6d85e519484fa0`. The standalone publication manifest is [`ARTIFACT_MANIFEST.json`](ARTIFACT_MANIFEST.json), and release-asset checksums are [`ARTIFACTS.sha256`](ARTIFACTS.sha256).

No required final-system binary artifact is missing. The release does not contain deterministic caches:

| Cache | Recorded bytes | Status |
| --- | ---: | --- |
| clean TRAIN cache | 363278219 | regenerate from pinned TRAIN inputs and frozen protocol |
| official-v6 cache | 77225389 | regenerate from pinned official inputs and frozen protocol |

The separate validation-only K-ary scaling checkpoints/cache are not part of the final-system artifact bundle. The included frozen report is sufficient to regenerate Figure 3.

## Artifact roles

- `results/official_v6/results.json`: authoritative official test artifact, byte-identical to the frozen result.
- `results/clean_v6_development/FINAL_RESULTS.json`: normalized publication copy of the clean-v6 development aggregate and direct parent provenance for official-v6.
- `results/clean_v6_development/seeds/*/PRE_DEVELOPMENT_FROZEN.json`: normalized publication copies of the three seed freezes.
- `results/kary_scaling/KARY_VALIDATION_REPORT.json`: validation-only matched-row K-ary scaling artifact used by Figure 3; it is not an official test.

## Normalized publication copies

Four pre-publication JSON files contained an irrelevant absolute workspace prefix. Only that prefix was removed, leaving relative original experiment paths under `5/`. No metric, seed, attack, range, model/data hash, or calculation changed.

| Publication file | Original artifact SHA-256 | Publication SHA-256 |
| --- | --- | --- |
| `results/clean_v6_development/FINAL_RESULTS.json` | `d49ed627bbdea3f8765c7e776cc3132dffaa07208ddcb069cd840e2d12aa9997` | `a879106602e3b9f31517e29ca6cfccf80f86bbfb20589d263bbaf0a66e9a04bb` |
| `results/clean_v6_development/seeds/424242/PRE_DEVELOPMENT_FROZEN.json` | `4466cd8b25b69c75ba53c0dffdc94ea7570bf82c0c16ccb11e4862d37744ba2a` | `33610c1784945864cc102d83f58a4095679cc72e105417be8f8d9d728437549b` |
| `results/clean_v6_development/seeds/271828/PRE_DEVELOPMENT_FROZEN.json` | `e4a1abad774fbdd535d3b15bddacf181bcd940f9263d789214735c4b69bac935` | `4666b6c263c54f71c5be0a78c37352ec6900d53eb5abfb535c80ad52d8be562c` |
| `results/clean_v6_development/seeds/314159/PRE_DEVELOPMENT_FROZEN.json` | `aa29f6eb26f1188f48d3dbf546c05599b2219a23556c48ad810e66edc5a3dc2c` | `4a510c25ad0287f1e38d9e0ab395beddc760c97b8254b08d466b95a79c07895c` |

Frozen marker and checksum files retain the original artifact hashes as provenance values; they do not claim that normalized publication copies are byte-identical. The active publication `configs/official_v6.yaml` and `official_test_v6_protocol.py` use publication-layout paths and the new publication hashes. Scientific protocol values remain unchanged.

## Figure publication copies

The final paper uses an in-LaTeX/TikZ architecture diagram and two external files. The generator was reduced to those external Figures 2 and 3, given neutral deterministic IDs, and assigned publication metadata:

- Author: `Mihnea-Teodor Stoica`
- Creator: `recursive-geodesic-composition/figures/generate_figures.py`
- SVG hash salt: `recursive-geodesic-composition-v1-paper`

Regenerated PNG pixels are identical to the pre-publication plots; only metadata/encoding identifiers changed where expected. Current publication hashes are:

| Figure | SHA-256 |
| --- | --- |
| `figures/fig2_official_per_corpus.png` | `e21c7c86fa86f9c4240f06f105cfe6fec5c5b2877758935a825a0498b5dea0fc` |
| `figures/fig2_official_per_corpus.svg` | `7984f57b567d6aa967ed283d7c8319bbc71080eff46ae19d0ab7b8b499852269` |
| `figures/fig3_kary_scaling.png` | `36699080d40cdf4ccf233a9decc37874843304793e948324b274096d1b9380f7` |
| `figures/fig3_kary_scaling.svg` | `867e3158deddf972184c200fff3a583b5e2b861d9182efdea2402d0c4979f418` |

Unused external figures and their figure-only frozen inputs were removed from the paper snapshot.

## AUC presentation rounding

For seed 271828, the authoritative JSON value is `0.5821244999999999`. The frozen report prints `0.582124`; the final paper displays `0.582125`. These are presentation variants of the same floating-point result, not different experiments. Frozen result/report files were not edited.

## Hash terminology

`reproducibility/hashes.txt` distinguishes original artifact hashes in comments from publication-copy verification lines. A publication-copy hash verifies the file in this snapshot. An original artifact hash remains a provenance value and implies byte identity only where explicitly stated.

## Publication code layout

`architecture.py` and `data.py` are byte-identical to their frozen source artifacts. Publication modules use package-relative imports and repository-relative CLI resolution. The active official config/protocol additionally uses publication-layout provenance paths and hashes for the four normalized JSON copies. Equations, tensor operations, masks, seeds, data ranges, tolerances, attacks, bootstrap definitions, and checkpoint selection were not changed.

Original experiment paths are listed in [`source_mapping.md`](source_mapping.md).
