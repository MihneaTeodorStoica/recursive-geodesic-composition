# Reproducing the supplement

Run commands from the root of a writable extracted supplement, using the paths shown (all are relative to that root). Keep the supplied snapshot immutable; the preparation, training, evaluation and audit namespaces are create-only. Use a separate writable copy for any new cache or result. This guide describes inspected commands and recorded artifacts; it does not claim that this machine downloaded data, trained a model, or reran a GPU evaluation.

## What is and is not included

The TMLR supplement upload limit is 100 MB. Six fitted parent-only probes (47,334,402 bytes total) are included under `weights/`; the remaining 60 released model/new-probe artifacts total 8,062,738,309 bytes and are hosted externally. [CHECKPOINT_ACCESS.md](CHECKPOINT_ACCESS.md) gives the public Hugging Face repository, immutable revision, group download commands and original-to-hosted mapping. The released 66-file collection includes 63 exact originals and three separately hashed metadata-only warm-state derivatives; no identifying warm original is anonymously redistributed. Teacher/dataset/task/embedding caches and nine separate older scaling/matched-pooling payloads remain unavailable or external.

The root helper is the offline first step:

```bash
python supplement.py verify
python supplement.py report
```

`verify` checks the size and SHA-256 of every file listed in `reproducibility/PACKAGE_MANIFEST.json`; `SHA256SUMS` and the package manifest are checksum containers, not self-authenticating signatures. `report` recomputes fidelity means and paired-contrast means from the retained 4,000 per-document safe-baseline scores, then checks 120 saved classifier AUC point estimates against 60 saved prediction files. It does not refit classifiers or replay bootstrap intervals. Neither command downloads a teacher/corpus, loads a model checkpoint, prepares a tensor cache, trains, runs model inference or benchmarks timing. A saved table/report or per-document scalar score is not a saved embedding or model-prediction tensor. JSON has per-document score records for some studies, but `.pt` leaf, parent and prediction tensors used by exact inference replays are not supplied. Thus this offline check is not a numerical model replay.

Materialize the six included probes into their protocol and lookup paths; both roles are repeatable selectors:

```bash
python supplement.py materialize weights \
  --role candidate_parent_only_fitted_probe \
  --role source_parent_only_fitted_probe \
  --verify-only
python supplement.py materialize weights \
  --role candidate_parent_only_fitted_probe \
  --role source_parent_only_fitted_probe \
  --mode hardlink
```

The helper verifies exact byte size and SHA-256 before placing a selected asset. Use `--mode copy` if hard links are unavailable. These probes do not replace final/source model checkpoints.

## Software and external requirements

The project requires CPython `>=3.12,<3.13`. `requirements.txt` installs the editable package plus `pytest`; runtime pins in `pyproject.toml` are `torch==2.13.0`, `transformers==5.14.1`, `datasets==5.0.0`, `numpy==2.5.1`, `scikit-learn==1.9.0`, `scipy==1.18.0`, `pandas==3.0.5`, `PyYAML==6.0.3`, `matplotlib==3.11.1`, `Pillow==12.3.0`, and `tensorboard>=2.14`. `uv.lock` records the Python 3.12 resolution and transitive package pins. The recorded official environment was CPython 3.12.14, PyTorch 2.13.0+cu130, NumPy 2.5.1, with CUDA visible; the publication-validation record reports a fresh `uv` environment and `uv pip check` passing for 96 installed packages. Packaging-only checks were performed under CPython 3.14.8, outside this project's supported range; no runtime model verification was done in that environment. The recorded run does not specify a GPU model, CUDA driver/toolkit, system image or every transitive system component. Treat a compatible CUDA driver and adequate device/storage capacity as external prerequisites; CPU fallback in some utility code is not a practical substitute for the training and full evaluation recipes below.

The documented installation recipes are:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or, using the recorded `uv` workflow:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Teacher and corpus preparation require network access to the pinned model and datasets, their hosting terms must be accepted, and no access token is included. Table-fragment rendering itself does not require a TeX distribution. This package has no manuscript build step.

## Frozen inputs and revisions

All final-v6, official-v6, safe-baseline and K-ary validation configs pin the same frozen teacher and four corpora. The final-v6 source-index ranges are TRAIN `[140000,152000)`, development `[152000,161000)`, an untouched buffer `[161000,162000)`, and official scan `[162000,192000)` for each corpus. Indices are source dataset indices (not the count of nonempty documents); the stream preparer filters empty text inside the declared range.

| Input | Dataset/config | Revision |
| --- | --- | --- |
| Teacher | `Alibaba-NLP/gte-modernbert-base` | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` / `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| FineWeb | `HuggingFaceFW/fineweb` / `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` |
| OpenWebText | `Skylion007/openwebtext` | `79d93d786212f7344586290adb811d4ae6a1762c` |
| English Wikipedia | `wikimedia/wikipedia` / `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |
| LongEmbed retrieval control | `dwzhu/LongEmbed` | `10039a580487dacecf79db69166e17ace3ede392` |

The active final configs override the legacy `FrozenTeacher` class default; use the configured GTE teacher above, not a class default. LongEmbed is an additional retrieval dataset, not one of the four training/official corpora. Reproducing that control also requires its pinned task data.

## Checkpoint directory, verification and loading

For public weights, first use `python scripts/download_checkpoints.py --group final --verify` and repeat `--group source --group historical_probes` for the official/source-comparison prerequisites. The helper pins Hugging Face revision `9187157fa4fc6f790be1775475952f14ae33d02e`, verifies every selected file, reuses matching local files and places the recorded protocol/lookup paths without authentication. All groups and manual downloads are in CHECKPOINT_ACCESS.md. The original `supplement.py materialize` workflow below remains available for a flat directory of exact original assets; it is not a sanitized-warm materializer. Original experiment manifests and provenance guards remain unchanged.

For example, after placing the exact authorized model files in a flat directory, verify and materialize the final checkpoints:

```bash
ARTIFACT_DIR=artifacts/local-model-weights
python supplement.py materialize "$ARTIFACT_DIR" \
  --role final_clean_v6_checkpoint --verify-only
python supplement.py materialize "$ARTIFACT_DIR" \
  --role final_clean_v6_checkpoint --mode hardlink
```

To materialize continuation initialization checkpoints from the externally supplied directory, select their exact role:

```bash
python supplement.py materialize "$ARTIFACT_DIR" \
  --role continuation_initialization_source_checkpoint --verify-only
python supplement.py materialize "$ARTIFACT_DIR" \
  --role continuation_initialization_source_checkpoint --mode hardlink
```

For official preaccess/evaluation, materialize the final and source model roles from the external directory and the candidate/source probe roles from `weights/`. Use `--mode copy` if hard links are unavailable. Never substitute another seed, checkpoint role or nearby model revision; the protocols verify local artifact hashes before use.

The checked-in model builder reconstructs dimensions from the checkpoint config and calls `load_state_dict(..., strict=True)`. This small loader prints no identifying paths and can be run after final checkpoint materialization:

```bash
python - <<'PY'
from pathlib import Path
import torch
from recursive_geodesic_composition.evaluate_kary_validation import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = Path("results_kary4_strict_autoencoder_v6_clean_development_seed424242/autoencoder_seed424242.pt")
model, checkpoint = build_model(path, device)
assert checkpoint["variant"] == "autoencoder"
assert model.d == 768 and model.hidden == 1024 and model.arity == 4
assert model.decoder_mode == "strict"
PY
```

A caller with ordered, already-computed unit leaf embeddings and exact positive token masses can run inference without the teacher/datasets, once a checkpoint is local. The tensors must have shapes `[N,768]` and `[N]`; masses are token counts, leaves remain in original contiguous order, and any inactive K4 slots are trailing zero-mass slots. For more than four leaves, `recursive_reduce` carries singleton remainders without merging a synthetic child:

```python
from recursive_geodesic_composition.architecture import recursive_reduce

leaves = torch.load("input/leaves.pt", map_location="cpu", weights_only=True)
counts = torch.load("input/token_counts.pt", map_location="cpu", weights_only=True)
with torch.inference_mode():
    root, levels = recursive_reduce(
        model,
        leaves.unsqueeze(0).to(device),
        counts.to(torch.float32).unsqueeze(0).to(device),
    )
    decoded_slots = model.decode(root)
```

`decoded_slots` is produced from the parent alone. The strict decoder receives no leaves, child labels or oracle token masses at prediction time. Do not interpret its output as a guarantee of semantic information recovery.

## Model and continuation recipe

The final model maps 768-dimensional teacher leaves to 768-dimensional parents using a 1024-hidden, 8-layer, 16-head K4 merger (FFN width 3072). It normalizes children, uses token masses in the weighted anchor, and applies a bounded tangent exponential-map correction with maximum angle 0.5 rad. Reduction uses deterministic contiguous K4 groups; K=2, 3 or 4 leading children are valid and trailing slots are masked. The strict one-layer decoder is parent-only. Scientific code and its existing training objectives are not modified by this guide.

The three published final checkpoints are not trained from scratch. Each continues from a seed-matched source checkpoint: source decoder weights are discarded, the strict decoder is freshly initialized, the decoder-only stage runs for 300 updates, then encoder and decoder run jointly for 100 updates with encoder learning-rate scale 0.02. The optimizer restarts. The final checkpoint is frozen at step 400 without validation-based checkpoint selection. The exact source checkpoints are now publicly downloadable through the pinned `source` group; the historical training run that created those sources is still not established by this supplement. Repeating the continuation recipe therefore requires those exact source files and regenerating the pinned TRAIN cache; random initialization would be a different experiment.

The exact clean-v6 sequence is one-shot and requires a fresh writable workspace. The source checkpoint role must already be materialized. Preparation streams the pinned TRAIN ranges and downloads the teacher; the recorded TRAIN cache was 363,278,219 bytes and preparation requires at least 1,800,000,000 free bytes. Configs use leaf/target batches of 32, training batch size 8 and a 32,768-token teacher-padding budget. They guard at least 805,306,368 free bytes before training and 536,870,912 before checkpoint writes, with an estimated 500,000,000 bytes per checkpoint.

```bash
python scripts/final_v6/prepare_train.py \
  --config configs/final_v6/seed424242.yaml --device cuda

for seed in 424242 271828 314159; do
  python scripts/train_final_v6.py \
    --config "configs/final_v6/seed${seed}.yaml" \
    --variant autoencoder --seed "$seed" --device cuda
done

# Freeze all three checkpoints and TRAIN-only probes before any development access.
python scripts/final_v6/freeze_training_and_probes.py --device cuda
python scripts/final_v6/prepare_development.py \
  --config configs/final_v6/seed424242.yaml --device cuda
for seed in 424242 271828 314159; do
  python scripts/final_v6/evaluate_development.py \
    --config "configs/final_v6/seed${seed}.yaml" --device cuda
done
python scripts/final_v6/finalize.py
```

The order is substantive: freeze all seed checkpoints/probes before preparing development data; then evaluate each frozen seed and finalize. The preparation scripts are networked, the training jobs are expensive GPU runs, and the cache/checkpoint/results namespaces refuse overwrites. No command above has been run for this guide.

## Official evaluation and its preserved recovery

Official-v6 scans `[162000,192000)` and takes the first 1,000 prefix-eligible documents per corpus (4,000 total), using the same documents for both arms. The safe teacher-comparable arm makes one direct K4 merge on a 1,025–4,096-token prefix and uses a direct teacher target. The unrestricted full-document arm covers all available tokens as contiguous leaves of at most 1,024 tokens: no truncation, token/leaf/depth cap, full-root teacher target, teacher cosine or AUC. The recorded official cache is 77,225,389 bytes; its frozen full-document profile has 9,142,043 tokens, 11,356 leaves and 4,789 internal merges across the 4,000 documents, with maxima of 58,731 tokens, 58 leaves and depth 3. These are retained-run accounting values, not newly computed here.

In a fresh workspace, materialize final/source model checkpoints and all official candidate/source probes, then run preaccess verification before any official cache/data access. Preparation requires network access to all pinned corpus streams and the teacher. Its evaluation batches are 16 leaves, 4 teacher targets, 64 safe roots, 64 full merges and 512 decoder items; storage reserves 536,870,912 bytes, budgets 1,048,576 bytes per full-document shard and requires 268,435,456 free bytes for results, failing rather than truncating. The first evaluator is retained specifically as an intended failure: it creates the one-shot start marker, then raises `KeyError: 'confirmation'` on stale provenance lookups before loading safe tensors or producing a metric. The frozen recovery repairs only those lookups and uses the prepared cache without rescanning the corpus.

```bash
# No corpus/cache access; checks local weights and frozen provenance only.
python scripts/official_v6/verify_preaccess.py
python scripts/prepare_official_v6.py \
  --config configs/official_v6.yaml --device cuda

# Historical expected failure. Run in this order only in a fresh official namespace;
# it intentionally leaves results_kary4_strict_autoencoder_official_test_v6.EVALUATION_STARTED.
python scripts/official_v6/evaluate_official_test_v6_initial.py \
  --config configs/official_v6.yaml --device cuda

# Run after that expected failure, with the complete cache and marker still present.
python scripts/evaluate_official_v6.py \
  --config configs/official_v6.yaml --device cuda
```

The recovery refuses to run without the spent initial marker, refuses an existing finalized output/recovery marker, and claims its own one-shot namespace. Do not run the initial failure in a workspace containing a current evaluation; do not try to repeat a failed preparation or recovery by deleting markers. Use a clean workspace for a fresh official replay. The existing `results/official_v6/results.json` and `REPORT.md` are frozen results, not outputs of these commands on the current machine.

## Saved rendering and table/figure map

Use the root `supplement.py report` for the package's offline saved-summary output. The following renderers are separate from inference: they consume tracked results, manifests and JSON per-document scalar scores, not freshly generated model vectors. They write generated presentation files, so run them only in a writable working copy when regenerating those files.

```bash
# Supporting official/scaling plots (legacy filenames; not current manuscript figure numbering).
python figures/generate_figures.py

# Table 4/report from saved safe-baseline summaries only (the `report` stage).
python scripts/evaluate_safe_baselines.py report \
  --config configs/safe_baselines_v1.yaml

# Tables 8 and 10 and the recursive-reuse plot/report.
python scripts/render_recursive_reuse.py \
  --config configs/recursive_reuse_v1.yaml

# Table 13 and the reconstruction-calibration plot/report.
python scripts/render_reconstruction_calibration.py \
  --config configs/reconstruction_calibration_v1.yaml

# Table 15 and the continuation comparison plot/report in manuscript presentation units.
python scripts/publish_continuation_comparison.py
```

The top-level figure generator reads only `results/official_v6/results.json` and `results/kary_scaling/KARY_VALIDATION_REPORT.json`; it does not train the K8 model or regenerate an official result. A separate packaging-copy regeneration produced identical RGBA pixels for those two supporting PNGs. The current manuscript's Figure 1 is inline TikZ; Figures 2–4 are the recursive-reuse, reconstruction-calibration and continuation plots under `results/`, as mapped in EVIDENCE_MAP.md. This archive has no manuscript build. The corresponding renderers consume saved scalar scores, not new model outputs. Table 4's retained fragment can likewise be rendered from saved summaries. Unlike `supplement.py`, these source renderers import project modules and therefore need the project's dependencies even for presentation-only use. Inline manuscript tables are mapped below to their result artifacts.

| Manuscript table | Frozen source retained in the supplement |
| --- | --- |
| 4 | `results/safe_baselines_v1/safe_baselines_table.tex` |
| 5 | `results/safe_baselines_v1/RESULTS.json` and `DOCUMENT_SCORES.json` |
| 6–7 | `results/official_v6/results.json` |
| 8 | `results/recursive_reuse_v1/recursive_reuse_table.tex` |
| 9 | `results/official_v6/results.json` cache provenance; additional population summaries in `results/recursive_reuse_v1/FULL_POPULATION.json` |
| 10 | `results/recursive_reuse_v1/full_population_table.tex` |
| 11–12 | `results/official_v6/results.json` |
| 13 | `results/reconstruction_calibration_v1/reconstruction_calibration_table.tex` |
| 14 | `results/v1_k4_matched_pooling/RESULTS.json` and `DOCUMENT_SCORES.json` |
| 15 | `results/continuation_comparison_v1/continuation_comparison_table.tex` |
| 16 | `results/longembed_review_v1/RESULTS.json`; aggregate `table.tex` is also retained |
| 17 | `results/kary_scaling/KARY_VALIDATION_REPORT.json` |

The five additive control outputs are retained at `results/input_order_v1/table.tex`, `results/tree_families_v1/table.tex`, `results/probe_families_v1/table.tex`, `results/longembed_review_v1/table.tex` and `results/decoder_gradient_v1/table.tex`. They have no separate offline renderer: the table/report files are frozen presentation outputs. `RESULTS.json` and saved per-document score arrays permit scalar summary inspection, not regeneration of unavailable embedding/prediction vectors. Re-rendering PNG/SVG can change bytes through library, font or metadata differences without changing the plotted values; run `supplement.py verify` against the final package manifest rather than assuming historical figure hashes remain current.

## Additive evaluation commands and their prerequisites

These analyses are not fresh confirmatory tests and reuse previously evaluated documents. They cannot run offline from tracked JSON alone: they need local `.pt` caches/vectors, exact weights, and pinned network inputs for preparation/re-encoding. The following are inspected **individual source entrypoints**, not a turnkey sequential rerun from this ZIP. In a disposable working copy, move only the target study's supplied result directory to a separate evidence location before running a create-only stage. Preserve its frozen upstream dependencies. Some downstream configs pin the exact historical upstream result/inference-manifest bytes; rerunning an upstream stage changes timestamps, environment and hashes, so a regenerated upstream run cannot automatically replace that historical input. Exact omitted upstream tensors, or a separately declared fresh-run provenance freeze with unchanged scientific settings, are needed to exercise those chains. Do not bypass guards or relabel a fresh freeze as the historical experiment.

```bash
# Safe baseline cache (network + teacher), GPU inference/classifier analysis, scalar audit.
python scripts/prepare_safe_baselines.py \
  --config configs/safe_baselines_v1.yaml --device cuda
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python scripts/evaluate_safe_baselines.py all \
    --config configs/safe_baselines_v1.yaml --device cuda
python scripts/audit_safe_baselines.py \
  --config configs/safe_baselines_v1.yaml

# Recursive reuse (reuses the safe cache/vectors and final checkpoints).
python scripts/evaluate_recursive_reuse.py all \
  --config configs/recursive_reuse_v1.yaml --device cuda
python scripts/audit_recursive_reuse.py \
  --config configs/recursive_reuse_v1.yaml --device cuda

# Parent-copy/decoder/probe reconstruction calibration.
python scripts/evaluate_reconstruction_calibration.py all \
  --config configs/reconstruction_calibration_v1.yaml --device cuda
python scripts/audit_reconstruction_calibration.py \
  --config configs/reconstruction_calibration_v1.yaml --device cuda

# Source-vs-final continuation comparison; source classifier fitting is new.
python scripts/evaluate_continuation_comparison.py all \
  --config configs/continuation_comparison_v1.yaml --device cuda
python scripts/audit_continuation_comparison.py \
  --config configs/continuation_comparison_v1.yaml --device cuda
python scripts/check_continuation_numerics.py --device cuda
```

The `all` stage performs fresh inference and analysis; audits replay models and independently recompute scores. The first additive analysis also regenerates a safe-only teacher cache (network required). Recursive reuse/reconstruction/continuation require the prior safe-baseline and reconstruction caches/tensors; continuation additionally requires both source/final checkpoints and six source/candidate probes. The saved `.json` scalar score and classifier files do not replace the missing `.pt` tensors. Their audits are therefore unavailable from the ZIP alone.

The remaining controls each have a fixed no-argument entrypoint (except the final audit's optional positional study list):

```bash
# Requires the safe-baseline inputs and exact final checkpoints/probes.
python scripts/evaluate_input_order.py
python scripts/evaluate_tree_families.py

# First prepare the separate review TRAIN cache; requires the safe-baseline cache/output,
# pinned source data and teacher.
python scripts/prepare_review_train.py
python scripts/evaluate_probe_families.py
python scripts/evaluate_decoder_gradient.py

# Requires the pinned LongEmbed task data and final checkpoints.
python scripts/evaluate_longembed_review.py

# Replays all five controls; requires their generated tensor/model artifacts.
python scripts/audit_review_completion.py
```

These are separate additive controls, not a reproduction of historical source-model training. The order study uses all 24 permutations on 606 exact-four prefixes; the tree study also re-encodes 512- and 256-token leaves with the teacher; probe families and the decoder-gradient intervention train new models/probes on fresh TRAIN data; LongEmbed uses four fixed tasks. The decoder-gradient run is a new 300 decoder-only + 100 joint-step intervention, not a historical training recipe. `audit_review_completion.py` checks/replays those outputs and requires their omitted `vectors.pt`, `predictions.pt` and related artifacts. The K=8 report is already stored; Figure 3 rendering does not require rerunning the separate K=8 validation training/preparation.

## Numerical and packaging caveats

The official pooled bootstrap and its per-corpus results are conditional on the three fixed checkpoints; they do not estimate an arbitrary training-seed population. The unrestricted full-document arm intentionally has no teacher target/cosine/AUC. Its 1,024-token leaf size, unlimited document coverage and singleton-carry K4 reduction are protocol constraints, not tunable limits. The safe arm's 4,096-token teacher-target/prefix bound is not a full-document result. Across the training recipe, chunks are at most 1,024 tokens, teacher targets are bounded by the configured 6,144-token maximum, training trees allow up to 96 leaves, and mixed precision is disabled. The official path stores full-arm leaf embeddings in FP16 and computes the safe arm/model inference in FP32 as configured.

Regenerated teacher embeddings/cache bytes are not promised to be identical to the historical cache, even with the same document identities and pinned revisions. Dataset streaming behavior, library/CUDA versions and hardware can change floating-point outputs or timing; the protocol reports all outcomes and keeps the same frozen populations rather than selecting on a new result. The archived audit verdicts and JSON metrics are historical records unless a local rerun is explicitly reported.

Finally, package anonymization can affect immutable byte-hash checks. Source metadata, path strings, image metadata, release manifests and packaging-specific code may change bytes while numeric JSON and executable scientific operations remain unchanged. Historical artifact hashes describe their original bytes; they are not proof that the anonymized file is byte-identical. Active package hashes are rebound to the sanitized snapshot and checked by `supplement.py verify`. If an older script encounters a pre-packaging hash reference, do not disable that check or edit a metric: first establish whether the reference is historical or active against the packaged manifest. No Git history or historical commit identifier is required by the anonymous supplement.

The rebuild independently investigated all 19 existing-path source-hash mismatches. [HASH_DISCREPANCY_AUDIT.md](HASH_DISCREPANCY_AUDIT.md) classifies 15 table/presentation-serialization differences, three documented historical recovery versions and one unresolved preparer fingerprint; it records every expected/current digest and its consumer impact. Five exact identity-free historical source/configuration copies are retained as non-active provenance references. No historical source hash was silently replaced or represented as resolved without evidence. Strict full-vector audits can still reject these fingerprints even after omitted tensors are restored. Do not bypass guards or rewrite scientific results; a fresh analysis needs its own coherently frozen provenance chain. The package-level helper remains the offline integrity/saved-scalar path.