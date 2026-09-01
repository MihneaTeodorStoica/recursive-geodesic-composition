# V2 CLS overnight run log

This file is append-only.

## 2026-04-26 — protocol amendment V2.1 (before retraining)

- Decision follows the committed seed-314159 TRAIN-only saturation diagnostic (`8b0058c`), conclusion B.
- Historical V2.0 runs are preserved; no checkpoint is resumed or overwritten.
- V2.1 changes only the Stage-A/B/C geodesic health stop: preserve telemetry threshold `angle >= 0.95 * 0.5 = 0.475` rad but require a nonzero saturation fraction for **three consecutive optimizer steps** before rejecting. This is applied identically to all three new fresh-from-random seeds.
- V2.1 adds practical event-level per-merge diagnostics: raw projected-tangent norm, bounded correction norm/angle, active arity, saturation flag, plus sampled batch source row/document identities. Exact row-to-merge correspondence is not available for batched/recurrent grouping and is explicitly not claimed.
- No development, buffer, official, or downstream source opened.
