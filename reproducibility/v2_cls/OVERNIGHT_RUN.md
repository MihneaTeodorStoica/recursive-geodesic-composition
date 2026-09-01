# V2 CLS overnight run log

This file is append-only.

## 2026-04-26 — protocol amendment V2.1 (before retraining)

- Decision follows the committed seed-314159 TRAIN-only saturation diagnostic (`8b0058c`), conclusion B.
- Historical V2.0 runs are preserved; no checkpoint is resumed or overwritten.
- V2.1 changes only the Stage-A/B/C geodesic health stop: preserve telemetry threshold `angle >= 0.95 * 0.5 = 0.475` rad but require a nonzero saturation fraction for **three consecutive optimizer steps** before rejecting. This is applied identically to all three new fresh-from-random seeds.
- V2.1 adds practical event-level per-merge diagnostics: raw projected-tangent norm, bounded correction norm/angle, active arity, saturation flag, plus sampled batch source row/document identities. Exact row-to-merge correspondence is not available for batched/recurrent grouping and is explicitly not claimed.
- No development, buffer, official, or downstream source opened.

## 2026-04-26 — V2.1 TRAIN execution

- Stage A completed for all three seeds under patience=3. Seed 314159 reproduced its isolated step-276 event, logged it, did not persist, and completed step 1200.
- Stage B and Stage C completed for 424242 and 271828.
- Seed 314159's original Stage-C computation reached step 1200 but atomic checkpoint serialization failed with `ENOSPC`; original JSONL and console log are retained. This is an engineering/storage failure, not a model failure.
- Historical V2.0 binary directories were moved intact to `/root/v2_cls_archive/` and replaced by same-path symlinks, freeing workspace volume. No artifact was altered.
- Stage C 314159 was rerun from its exact Stage-B checkpoint using an explicitly documented retry config with only a new output directory; it reproduced the same final logged scalar values and checkpointed successfully.
- Final V2.1 TRAIN-only 128-row audit cosines: 424242=.9818943, 271828=.9820268, 314159=.9822874. All roots finite/unit-normalized, effective ranks 85.84/85.85/86.14, no audited saturation.

## 2026-04-26 — V2 development access

- Development evaluator was frozen/committed before opening `[204000,213000)` and development access timestamp recorded.
- A cache of first 100 source-order safe direct-target documents per corpus was prepared with CLS leaves/direct CLS targets. The first evaluation attempt found a CPU/GPU decoder comparison bug before producing a metric; fixed evaluator was committed and the already-prepared cache was evaluated without a source rescan.
- Development safe teacher cosine mean across seeds: 0.98816099. Per seed: 424242=.98806107, 271828=.98815286, 314159=.98826903. Strongest logistic orientation-safe AUC: .5808375/.579425/.57245.

## 2026-04-26 — official preaccess audit

- Official range `[214000,244000)` remains unopened.
- Audited V2 closure and found no V2 TRAIN-fitted probes/floors and no complete V2 official two-arm evaluator. The limited development evaluator is not semantically sufficient for one-shot official use.
- Recorded `OFFICIAL_PREACCESS_BLOCKER.md` and stopped before official/downstream work rather than access a protected range with ambiguous/incomplete evaluator semantics.
