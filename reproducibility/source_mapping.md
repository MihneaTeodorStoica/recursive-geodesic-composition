# Source mapping

Original experiment paths are recorded by frozen experiment commit `32281e4a6836524448a61d87c5b4ed7eff32c0d3`.

## Publication metadata

| Publication file | Original experiment path | Notes |
| --- | --- | --- |
| `README.md` | — | Publication status, usage, and final paper facts |
| `LICENSE` | — | Conservative all-rights-reserved notice |
| `CITATION.cff` | — | Final title and author metadata |
| `requirements.txt` | `5/requirements.txt`; `5/paper_figures/requirements.txt` | Editable-install/test entry |
| `pyproject.toml` | `5/requirements.txt`; `5/paper_figures/requirements.txt` | Combined runtime/figure dependency pins |
| `.gitignore` | development root and `5/.gitignore` | Publication-specific union excluding large run artifacts |
| `reproducibility/ARTIFACT_MANIFEST.json` | recovered binary-artifact manifest | Standalone publication release mapping |
| `reproducibility/ARTIFACTS.sha256` | recovered binary-artifact manifest | Flattened GitHub Release asset checksums |

## Implementation

`architecture.py` and `data.py` are byte-identical to their frozen source artifacts. Other modules use package-relative imports and publication-root CLI paths.

| Publication file | Original experiment path | Role |
| --- | --- | --- |
| `src/recursive_geodesic_composition/__init__.py` | — | Publication exports |
| `src/recursive_geodesic_composition/architecture.py` | `5/architecture.py` | K-ary merger, tangent expmap, strict decoder, recursive reduction |
| `src/recursive_geodesic_composition/data.py` | `5/data.py` | Frozen teacher and streaming/tokenization utilities |
| `src/recursive_geodesic_composition/evaluate.py` | `5/evaluate.py` | Base grouped attack evaluation |
| `src/recursive_geodesic_composition/train.py` | `5/train.py` | Final clean-v6 trainer |
| `src/recursive_geodesic_composition/prepare.py` | `5/prepare.py` | Base cache preparation |
| `src/recursive_geodesic_composition/prepare_kary.py` | `5/prepare_kary.py` | Exact K-ary target/cache preparation |
| `src/recursive_geodesic_composition/evaluate_kary_validation.py` | `5/evaluate_kary_validation.py` | K-ary evaluator and checkpoint model builder |
| `src/recursive_geodesic_composition/information_retention.py` | `5/information_retention.py` | Probe classes/helpers in the official dependency closure |
| `src/recursive_geodesic_composition/prepare_retention_long.py` | `5/prepare_retention_long.py` | Supporting validation utility |
| `src/recursive_geodesic_composition/evaluate_retention_long.py` | `5/evaluate_retention_long.py` | Supporting validation utility |
| `src/recursive_geodesic_composition/validation_output_calibration.py` | `5/validation_output_calibration.py` | Diagnostics and orientation-safe AUC dependency |
| `src/recursive_geodesic_composition/deployment_regime_study.py` | `5/deployment_regime_study.py` | Fixed-attack/root helpers used by clean-v6 evaluation |

## Entry points and protocols

| Publication file | Original experiment path | Notes |
| --- | --- | --- |
| `scripts/train_final_v6.py` | thin publication wrapper around `5/train.py` | Training entry point |
| `scripts/prepare_kary_scaling.py` | thin publication wrapper around `5/prepare_kary.py` | K-ary preparation |
| `scripts/evaluate_kary_scaling.py` | thin publication wrapper around `5/evaluate_kary_validation.py` | K-ary evaluation |
| `scripts/evaluate_retention.py` | thin publication wrapper | Supporting validation utility |
| `scripts/prepare_retention_long.py` | thin publication wrapper | Supporting validation utility |
| `scripts/evaluate_retention_long.py` | thin publication wrapper | Supporting validation utility |
| `scripts/prepare_official_v6.py` | thin publication wrapper | Official cache preparation |
| `scripts/evaluate_official_v6.py` | thin publication wrapper | Metric-producing frozen recovery |
| `scripts/materialize_release_artifacts.py` | — | New release verification/path-placement helper |
| `scripts/final_v6/protocol.py` | `5/experiments/kary4_strict_autoencoder_v6_clean_development/protocol.py` | Final clean-v6 fail-closed protocol |
| `scripts/final_v6/prepare_train.py` | corresponding `5/experiments/.../prepare_train.py` | Fresh TRAIN-only cache preparation |
| `scripts/final_v6/prepare_development.py` | corresponding `5/experiments/.../prepare_development.py` | Fresh development cache preparation |
| `scripts/final_v6/freeze_training_and_probes.py` | corresponding `5/experiments/.../freeze_training_and_probes.py` | Pre-development checkpoint/probe freeze |
| `scripts/final_v6/evaluate_development.py` | corresponding `5/experiments/.../evaluate_development.py` | Development evaluator |
| `scripts/final_v6/finalize.py` | corresponding `5/experiments/.../finalize.py` | Aggregate finalizer |
| `scripts/official_v6/official_test_v6_protocol.py` | `5/experiments/kary4_strict_autoencoder_official_test_v6/official_test_v6_protocol.py` | Publication paths/hashes adjusted for normalized copies |
| `scripts/official_v6/prepare_official_test_v6.py` | corresponding `5/experiments/.../prepare_official_test_v6.py` | Safe/full cache preparer |
| `scripts/official_v6/evaluate_official_test_v6_initial.py` | `5/experiments/kary4_strict_autoencoder_official_test_v6/evaluate_official_test_v6.py` | Preserved expected pre-metric failure |
| `scripts/official_v6/evaluate_official_test_v6_recovery.py` | `5/experiments/kary4_strict_autoencoder_official_test_v6/evaluate_official_test_v6_recovery.py` | Authoritative result-producing evaluator |
| `scripts/official_v6/verify_preaccess.py` | corresponding `5/experiments/.../verify_preaccess.py` | Corpus/cache-free immutable preflight |

## Configurations

The three final-v6 seed configs and attack/K-ary/supporting validation configs are byte-identical to frozen source artifacts. The official config changes only publication-layout paths and hashes for normalized provenance JSON; all scientific values remain identical.

| Publication file | Original experiment path |
| --- | --- |
| `configs/final_v6/seed424242.yaml` | `5/experiments/kary4_strict_autoencoder_v6_clean_development/configs_train_availability_recovery/seed424242.yaml` |
| `configs/final_v6/seed271828.yaml` | `5/experiments/kary4_strict_autoencoder_v6_clean_development/configs_train_availability_recovery/seed271828.yaml` |
| `configs/final_v6/seed314159.yaml` | `5/experiments/kary4_strict_autoencoder_v6_clean_development/configs_train_availability_recovery/seed314159.yaml` |
| `configs/official_v6.yaml` | `5/experiments/kary4_strict_autoencoder_official_test_v6/config.yaml` |
| `configs/evaluator_attacks_v1.yaml` | `5/configs/evaluator_attacks_v1.yaml` |
| `configs/kary_scaling.yaml` | `5/experiments/kary8_leaf768_validation_v1/configs/kary8_leaf768.yaml` |
| `configs/kary4_reference.yaml` | `5/experiments/kary4_leaf510_vs1024_v1/configs/leaf1024.yaml` |
| `configs/information_retention_validation.yaml` | `5/configs/information_retention_validation.yaml` |
| `configs/multicorpus_generalization_v1.yaml` | `5/configs/multicorpus_generalization_v1.yaml` |

## Frozen results

Files not marked normalized remain byte-identical to frozen experiment artifacts.

| Publication file | Original experiment path | Notes |
| --- | --- | --- |
| `results/official_v6/results.json` | `5/results_kary4_strict_autoencoder_official_test_v6/results.json` | Authoritative official-v6 artifact |
| `results/official_v6/REPORT.md` | `5/results_kary4_strict_autoencoder_official_test_v6/REPORT.md` | Frozen generated report |
| `results/official_v6/ORIGINAL_SHA256SUMS` | `5/results_kary4_strict_autoencoder_official_test_v6/SHA256SUMS` | Original recorded hashes |
| `results/clean_v6_development/FINAL_RESULTS.json` | `5/results_kary4_strict_autoencoder_v6_clean_development/FINAL_RESULTS.json` | Normalized path-only publication copy |
| `results/clean_v6_development/seeds/*/PRE_DEVELOPMENT_FROZEN.json` | corresponding per-seed result directories | Normalized path-only publication copies |
| all other `results/clean_v6_development/` files | corresponding aggregate/per-seed result directories | Frozen experiment records; embedded original hashes remain provenance values |
| all `results/kary_scaling/` files | `5/results_kary8_leaf768_validation_v1/` | Validation-only Figure 3 source and provenance |

The prior external-figure-only result copies and the earlier retention-figure result table copies were removed because the final paper does not use them. Supporting code remains where it is part of the evaluator dependency closure.

## Final figures

| Publication file | Original experiment path | Notes |
| --- | --- | --- |
| `figures/generate_figures.py` | `5/paper_figures/generate_figures.py` | Trimmed to final external Figures 2 and 3; neutral publication metadata |
| `figures/fig2_official_per_corpus.png` | `5/paper_figures/fig2_official_per_corpus.png` | Regenerated publication copy; pixels unchanged |
| `figures/fig2_official_per_corpus.svg` | `5/paper_figures/fig2_official_per_corpus.svg` | Regenerated publication copy |
| `figures/fig3_kary_scaling.png` | `5/paper_figures/fig3_kary_scaling.png` | Regenerated publication copy; pixels unchanged |
| `figures/fig3_kary_scaling.svg` | `5/paper_figures/fig3_kary_scaling.svg` | Regenerated publication copy |

Figure 1 is generated directly in the final paper's LaTeX/TikZ source and has no external image in this repository.

## Frozen protocol records and tests

Files under `reproducibility/protocols/final_v6/` and `reproducibility/protocols/official_v6/` map to the same filenames under their respective `5/experiments/` directories. They are archival declarations, hashes, and launchers; publication-layout entry points are under `scripts/`.

Frozen protocol tests use publication imports/layout under `tests/`. `tests/test_publication_smoke.py` verifies final facts, result/figure hashes, the release manifest, and synthetic execution. `provenance.md`, `validation.md`, `hashes.txt`, and this mapping are publication-specific.
