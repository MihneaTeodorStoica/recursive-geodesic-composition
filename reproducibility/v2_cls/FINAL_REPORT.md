# V2 CLS report — TRAIN-only fresh-from-random run

## Stop status

**STOPPED before V2 development, official, or downstream access.** The fresh protocol was implemented and piloted on V2 TRAIN only, then frozen before full training. Two seeds completed all three stages; seed `314159` was permanently rejected at Stage A by the predeclared geodesic-saturation stop. Therefore there are not three final V2 checkpoints and these models are **not ready for a separately approved V2 development evaluation**.

This is not a reportable V2 performance claim. All reported numbers below are training-only engineering diagnostics on `[192000,204000)`.

## Why V2 changed

The inherited `FrozenTeacher` used masked mean pooling, while GTE ModernBERT documents normalized CLS embedding. V2 adds explicit pooling controls, preserving `masked_mean` as the legacy default for v1 reproduction and requiring `cls` in every V2 config/cache/checkpoint provenance chain. The CLS path adds normal special tokens to raw stored content IDs, uses `outputs.last_hidden_state[:,0]`, float32 casts and L2-normalizes.

Direct reference validation passed before cache/training: `reproducibility/v2_cls/CLS_REFERENCE_VALIDATION.json` has max absolute difference `0.0` on four texts against direct reference-style GTE usage.

## Reconstructed versus newly specified training procedure

`reproducibility/v2_cls/FRESH_TRAINING_PLAN.md` is the predeclared detail.

Inherited from published code/configuration: final 768/1024/8-layer/16-head K=4 strict architecture; contiguous K=4 reduction/singleton carry; raw token masses; 1024 and random 512--1024 leaf chunking; max teacher span 6144; four pinned corpora/revisions and equal mixture; semantic/tangent-expmap regularizers; direct teacher cosine, strict reconstruction/cycle, moment matching, AdamW/cosine and finite guards; seeds `424242`, `271828`, `314159`.

New because source-lineage information is missing: Stage A fresh merger-only training for 1200 steps at LR `1e-4`; Stage B 300-step decoder-only warm start from **the same V2 Stage-A checkpoint** at LR `2e-5`; Stage C 1200-step all-parameter training from the exact V2 Stage-B autoencoder state at LR `1e-4`. Stage C uses a checked full-autoencoder continuation implementation with mandatory matching explicit CLS provenance and a restarted optimizer. Fixed Stage-C step 1200 was predeclared as the sole final selection rule.

No learned v1 checkpoint was accessed for initialization.

## Pilot

Seed `424242`, V2 TRAIN cache only, shortened structurally identical A/B/C `120/40/120` steps:

- Stage A fixed-row matched teacher cosine: `0.973206 -> 0.973723`.
- Stage B reconstruction (logged first/last): `0.074078 -> 0.066087`.
- Stage C fixed-row matched teacher cosine: `0.979051` after joint training; reconstruction `0.009798`.
- All inspected values finite; root norms `[0.9999998, 1.0000002]`; no geodesic or semantic-correction saturation; root effective rank `85.68` on 128 fixed TRAIN rows.
- Pilot stage wall times: A 29 s, B 17 s, C 31 s.

The pilot passed the declared implementation sanity criteria. Raw logs/checkpoints remain under ignored `results/v2_cls/pilot_seed424242/`; concise log summaries are in `reproducibility/v2_cls/TRAINING_LOG_SUMMARY.json`.

## Full frozen-run outcome

### Seed 424242 — completed

- Stage A checkpoint: `8469baf70f3082955ce3bf1525434c6e9ca53c362f07db03dc4b1c40d6279435`
- Stage B checkpoint: `2e61098fb7de53d48849c6f9a43f0c4c9551e8966b9e46f60e893c1eb88c7837`
- Final Stage C checkpoint: `5a9641ad79d7d8631aa7ac07c0201d935bf3d8e0b2b407031c1c0a496e61e893`
- Fixed TRAIN-row Stage-A initialization/final cosine: `0.973206 -> 0.981894` (final Stage-C diagnostic).
- Final fixed-row teacher loss `0.018106`; reconstruction `0.010115`; cycle `2.19e-7`; effective rank `85.84`; max audited angle `0.05665`; no audited saturation.
- Wall time: A 174 s, B 43 s, C 187 s.

### Seed 271828 — completed

- Stage A checkpoint: `4b1c2bbab5eaa20aaaf7b8f90661b6eee8f5ddfbe9fd581ddecd24cce04e294f`
- Stage B checkpoint: `46bc70ca8905c1b09b931e92c692d8450a95ba944348d90d40de88f30943f787`
- Final Stage C checkpoint: `e126c442d47748c942e1c2074daccf356025c51a2dc64af72e11a0e6183dca04`
- Fixed TRAIN-row Stage-A initialization/final cosine: `0.973206 -> 0.982027`.
- Final fixed-row teacher loss `0.017973`; reconstruction `0.010137`; cycle `2.26e-7`; effective rank `85.85`; max audited angle `0.05199`; no audited saturation.
- Wall time: A 174 s, B 40 s, C 186 s.

### Seed 314159 — failed/preserved

- Stage A stopped at step 276/1200.
- Predeclared hard stop: `geodesic_saturation_nonzero_for_1_steps:0.111111112`.
- Last observed angle max `0.49999988` (the configured 0.5 rad bound); no Stage-A checkpoint was selected and Stages B/C were not run.
- Wall time before stop: 48 s.
- No restart, cherry-pick, seed replacement, or post-hoc hyperparameter change was performed.

Total full-run process time was 852 s (14 m 12 s), excluding cache construction; pilot process time was 77 s. Training log first/last values are machine-readable in `reproducibility/v2_cls/TRAINING_LOG_SUMMARY.json`.

## TRAIN-only provenance/audit

`reproducibility/v2_cls/FINAL_TRAIN_AUDIT.json` records:

- V2 TRAIN cache manifest SHA-256 `5d97c2742663c47ad3d01a3f5ae6f6e93c7616984c3a81ab14098560f10af73d`;
- explicit CLS teacher/revision in cache rows and completed checkpoints;
- 25,555 cached rows, 51,401,940 raw content tokens, and target-span token-accounting checks;
- completed checkpoint identities/selected step 1200 and Stage-B ancestry;
- `all_three_ready: false`.

The final code/config plan was frozen before full training in `reproducibility/v2_cls/FRESH_PROTOCOL_FREEZE.json`. Bound stage configs record exact V2 source checkpoint hashes. No V2 development, official, or downstream source was opened.

## Remaining concerns and decision

The fresh procedure demonstrates that the CLS cache, merger-first staging, strict decoder warm start, and full same-space continuation run stably for two seeds. It does **not** establish a viable three-seed protocol because the third declared seed saturated under the predeclared hard criterion. The correct next action is not development evaluation: preserve and analyze this TRAIN-only failure, then obtain separate approval for any new predeclared robustness protocol. No `v2-paper` release or PR should be made from these incomplete results.


## V2.1 amendment outcome

V2.1 (three-consecutive-step saturation persistence) completed all three fresh seeds. Final V2.1 checkpoint hashes are `c2464e5f884887779f35f4dff193c25ec1b9d80dbc91eae7263610b9e8d55c19` (424242), `0ced20f8619d46fb2c8be8acc6bcd7c751bdbf3d693885f500ed961a117c4ee8` (271828), and `0be4a8059b8e7673d563eada614ca95651c309bad07f6176b3919235f5253d27` (314159; documented ENOSPC checkpoint-write recovery). This is still TRAIN-only; no V2 development/official/downstream result is claimed here.


## V2.1 development (safe direct CLS spans)

Development `[204000,213000)` was accessed after evaluator freeze. A first GPU/CPU device-placement evaluator bug was corrected and recorded before any metric; no source rescan was performed. On 100 safe direct-target documents per corpus, three-seed mean CLS teacher cosine was `0.98816099`; per-seed strongest logistic orientation-safe AUCs were `.5808375`, `.579425`, `.57245`. Full machine-readable result: `reproducibility/v2_cls/DEVELOPMENT_SAFE_RESULTS.json`.


## Official preaccess status

Official V2 evaluation and downstream evaluation were not run. `OFFICIAL_PREACCESS_BLOCKER.md` documents the missing V2 TRAIN-fitted probe/floor and complete two-arm official evaluator closure. The official range remains unopened.
