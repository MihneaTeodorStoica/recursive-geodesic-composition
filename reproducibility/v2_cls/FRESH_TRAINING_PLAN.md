# V2 CLS fresh-from-random training plan

**Status:** pre-pilot; TRAIN-only. This plan authorizes only the already declared V2 TRAIN source-index interval `[192000,204000)` for each pinned corpus. It does not authorize reading V2 development `[204000,213000)`, V2 buffer `[213000,214000)`, V2 official `[214000,244000)`, or downstream benchmarks.

## Why this is a new protocol

Published clean-v6 is explicitly a continuation: 300 decoder-only updates and 100 joint updates from already learned source mergers. The source checkpoints' earlier training lineage is absent, and they must not be used for CLS. Consequently the clean-v6 300+100 schedule is not applicable to a random merger. This protocol is a **new fresh-from-random V2 training procedure**, not an exact rerun.

## Reconstructed components inherited directly from published code/configurations

- Final architecture: `GeodesicMerge`, 768-D input/output, hidden 1024, 8 merger layers, 16 heads, K=4 active prefix arities 2/3/4, strict parent-only four-slot decoder, semantic importance, tangent exponential-map correction, max angle 0.5.
- Reduction/chunking: exact contiguous K=4 reduction with singleton carry; raw content-token masses; fixed 1024-token leaves and deterministic random contiguous 512--1024 token leaves/spans, with train maximum 96 leaves.
- Targets: direct valid contiguous teacher spans only at `<=6144` content tokens; no invented target when longer.
- Teacher: `Alibaba-NLP/gte-modernbert-base` revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, `trust_remote_code: true`, special-token CLS then L2-normalized `last_hidden_state[:,0]`.
- Pinned equal mixture of FineWeb-Edu, FineWeb, OpenWebText, and English Wikipedia and their revisions from `configs/final_v6/*.yaml`.
- Existing objectives/machinery: direct parent cosine loss; strict decoder reconstruction; decoder cycle; mean/diagonal-variance distribution matching; semantic importance target/prior/delta/sharpness/slot terms; AdamW, gradient clipping 1.0, cosine schedule, deterministic Python/Torch seed, finite-gradient guards, and fixed-final TRAIN-only checkpoint writing.
- Earlier published from-scratch K=4/K=8 configs (`configs/kary4_reference.yaml`, `configs/kary_scaling.yaml`) supply the only published full-model-from-random duration/rate precedent: 1200 updates, AdamW LR `1e-4`, weight decay `.001`, cosine warmup 160 and floor `1e-5`, batch 8 or 16, and the same distribution/semantic regularizers.

## Newly specified choices (not recovered v1 settings)

1. **Stage A merger pretraining:** 1200 updates of the `encoder` variant from random initialization. It has no decoder and optimizes direct CLS teacher cosine plus published distribution and semantic regularizers. This makes the merger useful before decoder training.
2. **Stage B decoder warm start:** load only the matching Stage-A V2 encoder checkpoint (never v1); attach a newly initialized strict decoder; freeze the merger for 300 updates and optimize strict reconstruction/importance only. This reuses the published decoder warm-start mechanism, but its placement after Stage A is new.
3. **Stage C joint training:** unfreeze every parameter and train 1200 additional updates, with all Stage-A objectives plus reconstruction/cycle/strict-importance terms. The Stage-B optimizer is intentionally not resumed; Stage C begins a fresh AdamW/cosine schedule. This is a new, stated choice because no source lineage exists.
4. **Final selection:** fixed final Stage-C step 1200, independently for every declared seed, not a best training-loss checkpoint and not a development-selected checkpoint.
5. **No early performance selection:** numerical non-finiteness is a hard failure; a geodesic saturation event is recorded and permanently rejects that run while preserving logs/checkpoints. There is no development/official criterion and no seed replacement.

## Fixed final configurations

All final-stage values will be materialized in `configs/v2_cls/fresh_seed{424242,271828,314159}_{stage_a,stage_b,stage_c}.yaml` and their SHA-256 hashes frozen in `reproducibility/v2_cls/FRESH_PROTOCOL_FREEZE.json` before final seed training.

### Shared cache/data

- cache: `artifacts/v2_cls/cache_train_cls`, create-only;
- ranges: `[192000,204000)` only, 12,000 source rows scanned per corpus;
- four corpora equal weight `.25` each;
- 4 deterministic training augmentations; teacher max 6144; leaf/target batch 32; padded target budget 32768;
- batch size 8; no mixed precision.

### Stage A (new, merger-first)

- `variant=encoder`, random initialization, 1200 updates;
- LR `1e-4`, weight decay `.001`, cosine warmup 160, minimum LR `1e-5`;
- teacher coefficient 1; distribution mean `.25`, distribution variance `256`, minimum samples 8; semantic prior/delta/sharpness/slot `.01` each; no decoder/reconstruction/cycle;
- no validation cache, fixed final step 1200.

### Stage B (new sequencing; published decoder-only mechanism)

- `variant=autoencoder`, V2 Stage-A encoder initialization only, strict/predicted decoder, merger frozen;
- 300 updates; LR `2e-5`, weight decay `.001`, cosine warmup 50, floor `2e-6`;
- reconstruction `.15`, strict importance `.025`; teacher/merger/distribution/cycle disabled by existing frozen-encoder machinery;
- only full four-child rows are sampled; fixed final step 300.

### Stage C (new duration)

- `variant=autoencoder`, initialized from the V2 Stage-B checkpoint's encoder keys only, but this is prohibited by trainer semantics because it ignores Stage-B decoder. Therefore the final implementation must use a checked V2 full-autoencoder continuation preserving both Stage-B decoder and Stage-A merger state, with a fresh optimizer; this exact continuation code must be implemented and smoke-tested before pilot.
- 1200 updates; LR `1e-4`, weight decay `.001`, cosine warmup 160, floor `1e-5`;
- reconstruction `.15`, cycle `5.0`, distribution `.25`/`256`, importance `.025`, prior/delta/sharpness/slot `.01`, semantic importance, strict/predicted decoder;
- all parameters trainable from step 1; fixed final step 1200.

## Pilot (implementation sanity only)

Before the final freeze, one seed (`424242`) runs shortened but structurally identical stages: A=120, B=40, C=120. It uses the final TRAIN cache only and is not a hyperparameter search. It records initial vs final matched TRAIN parent cosine, teacher loss, output norms/finite fraction/effective rank, saturation fractions, recursion/token-mass checks, decoder reconstruction and memory/runtime. A pilot failure caused by an implementation defect may be fixed and rerun with preserved failed logs. A fundamental optimization failure may justify one documented TRAIN-only change before final freeze; no seed/lucky-run selection is permitted.

## Final acceptance for training readiness

The pilot may authorize final training only if: (1) all values/gradients/checkpoints finite; (2) Stage-A matched TRAIN cosine exceeds its initialization measurement; (3) no pathological geodesic or semantic-correction saturation (any nonzero saturation is recorded and reviewed); (4) recursive root norms are unit within tolerance and parent token masses exactly sum; (5) Stage-B reconstruction is finite and decreases from its first logged value; (6) hardware memory/runtime can complete the declared schedule. These are engineering sanity checks, not reportable generalization claims.

## Post-final TRAIN-only audit

After final Stage-C checkpoints for all three seeds, audit checkpoint/config/code/cache SHA-256s; row/checkpoint `teacher_pooling=cls`, teacher revision, seed and stage ancestry; zero v1 initialization; finite/unit-norm recursive outputs; noncollapse; exact token accounting; and raw logs/timing. Then stop before any development/official/downstream access.
