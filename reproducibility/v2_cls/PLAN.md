# V2 CLS predeclared plan

## Status and stop condition

This is a pre-development/pre-official V2 plan. No V2 development or official source range has been opened. Repository inspection found the final clean-v6 checkpoints were continuation-trained from released source checkpoints that were already learned models, and this publication repository does **not** contain the exact pre-final multi-stage lineage, configs, data ranges, or proof that those source parameters were never trained against the inherited masked-mean teacher space. Therefore V2 must not initialize from any v1 learned merger/decoder checkpoint. If exact pre-final lineage is required, this repository is insufficient; official V2 evaluation must stop until that lineage is recovered or this fresh protocol is explicitly accepted as a distinct V2 experiment.

## Branch/base

- Branch: `v2-paper-cls`
- Base commit: `c6b3f28830e7965533e1d036c2a749836b7848b6`
- v1 artifacts, results, official-v6 files, release assets, and downstream diagnostics are read-only and must not be modified.

## Teacher and embedding formula

- Teacher: `Alibaba-NLP/gte-modernbert-base`
- Revision: `e7f32e3c00f91d699e8c43b53106206bcc72bb22`
- `trust_remote_code: true`
- Pooling: `cls` only for V2.
- Formula: tokenize with the model's normal special tokens, run `AutoModel`, take `outputs.last_hidden_state[:, 0]`, cast to float32, and L2-normalize along the embedding dimension.
- Raw repository token chunks are still counted without special tokens; the CLS path adds CLS/SEP only at teacher forward time. Token masses remain raw content-token counts.
- Pre-training gate: `scripts/v2_cls/validate_cls_reference.py` must pass against the direct model-card-style reference on several texts. Current recorded pass: `reproducibility/v2_cls/CLS_REFERENCE_VALIDATION.json`.

## Architecture and reduction

Same deployable architecture as clean-v6 where applicable:

- embedding dimension 768;
- K=4 maximum arity with active arities 2, 3, and 4;
- deterministic contiguous K=4 reduction with singleton carry;
- normalized token-weighted anchor plus bounded tangent exponential-map correction;
- maximum geodesic angle 0.5 rad;
- hidden dimension 1024, 8 merger Transformer layers, 16 heads, FFN multiplier as in existing implementation;
- strict parent-only ordered four-slot decoder;
- semantic importance mode and auxiliary losses as declared in `configs/v2_cls/seed*.yaml`.

## Initialization and training schedule

Because exact pre-final v1 lineage is unavailable and v1 source checkpoints are contaminated/unknown with respect to masked-mean teacher supervision, V2 uses fresh random initialization for each declared seed and no learned v1 checkpoint:

- seeds: `424242`, `271828`, `314159`;
- configs: `configs/v2_cls/seed424242.yaml`, `seed271828.yaml`, `seed314159.yaml`;
- no `initialize_from_encoder_checkpoint` and no `initialize_from_encoder_checkpoint_sha256`;
- all parameters train jointly from step 1 (`freeze_encoder_steps: 0`, `autoencoder_objective_start_step: 0`, `autoencoder_objective_ramp_steps: 0`);
- 400 optimizer steps, batch size 8, learning rate `2e-5`, cosine schedule, 50 warmup steps, minimum LR `2e-6`, weight decay `0.001`;
- objectives/coefficient values otherwise follow the V2 config files, including reconstruction, cycle, distribution, importance, prior, delta, sharpness, and slot terms.

This is scientifically distinct from v1 clean-v6 continuation training and must not be reported as an exact v1 rerun.

## Data, ranges, and source separation

Pinned corpora/revisions are inherited from clean-v6 configs:

- `HuggingFaceFW/fineweb-edu`, `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- `HuggingFaceFW/fineweb`, `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`;
- `Skylion007/openwebtext`, revision `79d93d786212f7344586290adb811d4ae6a1762c`;
- `wikimedia/wikipedia`, `20231101.en`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.

Known spent/forbidden source-index ranges per corpus:

- official-v5: `[110000,140000)`;
- clean-v6 TRAIN: `[140000,152000)`;
- clean-v6 development: `[152000,161000)`;
- clean-v6 buffer: `[161000,162000)`;
- official-v6: `[162000,192000)`.

V2 declared ranges per corpus:

- TRAIN: `[192000,204000)`;
- development: `[204000,213000)`;
- buffer: `[213000,214000)`;
- future official one-shot candidate: `[214000,244000)`.

`python scripts/v2_cls/protocol.py` programmatically asserts non-overlap.

## Cache rules

V2 cache namespace: `artifacts/v2_cls/cache_train_cls` for TRAIN. Development and official caches must use V2-specific paths under `artifacts/v2_cls/` and must be create-only. Every cache row/sidecar/manifest must record teacher ID, revision, `teacher_pooling=cls`, tokenizer class, source range, chunking parameters, hashes, and code/config hashes. Training refuses a cache whose row-level teacher/pooling provenance mismatches the V2 config.

No v1 mean-pooled cache may be reused.

## Checkpoint selection and pre-official freeze

For this fresh protocol, training is TRAIN-only and writes the fixed step-400 checkpoint before development data is opened. Any development evaluator, probes, attack definitions, thresholds, bootstrap seed/replicates, and downstream code must be hash-recorded before development. Official data must not be opened until after models/probes/evaluator code and checkpoint selection are frozen.

## Development diagnostics

After all three checkpoints are frozen, run teacher-comparable development diagnostics analogous to v1, using only CLS teacher targets where a valid complete target exists:

- paired candidate-vs-direct-teacher cosine;
- generated-vs-teacher detectability with predefined attack families from `configs/evaluator_attacks_v1.yaml` unless a V2 attack config is frozen before development;
- strict decoder/constituent reconstruction;
- numerical health/collapse/rank checks;
- per-corpus analysis;
- bootstrap with seed `66031740` and 400 replicates unless changed before development.

Implementation bugs discovered on development may be fixed with full provenance and development rerun. Poor scientific performance is not a bug.

## Official protocol

Official source `[214000,244000)` remains unopened. If and only if the V2 protocol has been accepted and frozen, prepare a create-only one-shot official cache. Use the v1 conservative distinction: direct teacher fidelity only for valid complete CLS teacher targets, no fabricated/truncated long-span teacher target. Preserve safe teacher-comparable and unrestricted full-document evaluations separately.

## Probes and downstream

Any parent-only probes must be refit using V2 TRAIN data only; no v1 probes can be reused. Downstream retrieval must use CLS teacher embeddings for queries/direct references, exact CLS leaf embeddings for recursive and pooling baselines, and standard BEIR/pytrec_eval metrics. NFCorpus remains useful; at least one additional established retrieval dataset with substantial multi-leaf/depth>=2 documents must be selected based on length distribution before model evaluation. Malformed/empty records and qrel-referenced missing docs fail closed.

Classification and clustering tasks may run only after their dataset protocols validate in the CLS representation space.

## Audit requirements

Final audit must verify checkpoint/cache/result hashes, no v1 tracked/release artifact changes, source-range non-overlap, `teacher_pooling=cls` throughout V2 provenance, declared seeds/checkpoints, official access after freeze only, and an independent sample of documented CLS teacher embeddings. Final report path: `reproducibility/v2_cls/FINAL_REPORT.md`.
