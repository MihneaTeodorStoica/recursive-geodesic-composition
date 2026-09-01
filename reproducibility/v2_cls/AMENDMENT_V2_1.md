# V2.1 TRAIN protocol amendment: persistence-based geodesic stop

**Frozen before V2.1 training.** This amendment supersedes V2.0 only for a complete all-seed fresh rerun. V2.0 artifacts remain historical and unchanged.

## Change

All Stage A/B/C V2.1 configurations retain the 0.5-radian maximum and the existing saturation telemetry criterion `angle >= 0.95 * max_angle = 0.475`. They set `geodesic_saturation_patience_steps: 3`. The trainer rejects only after nonzero batch saturation on three consecutive optimizer steps. `reject_on_geodesic_saturation: true` and fail-fast behavior once the persistence condition is met are retained.

## Justification

The V2.0 seed-314159 diagnosis showed one saturated merge of nine at step 276, bounded/finiteness-preserving tangent-expmap behavior, no prior 16-step saturation, and no NaN/Inf. Clean-v6 did not use a hard saturation rejection; immediate termination was a V2.0 choice. The persistence rule is a strict safety screen for repeated behavior rather than an assertion that one bounded event is numerical invalidity.

## Uniform application

V2.1 restarts all three seeds (`424242`, `271828`, `314159`) from random Stage-A initialization and uses the unchanged three-stage A=1200/B=300/C=1200 schedule, data/cache, objectives, teacher CLS revision, architecture, and fixed final checkpoint rule. No V2.0 checkpoint may be resumed or used as initialization. A further amendment is allowed only if V2.1 demonstrates persistent instability and must again apply to all seeds from fresh initialization.

## Additional diagnostics

At a saturation event, the JSONL record now includes detached per-merge projected raw tangent norm, bounded correction norm/angle, active arity, saturation flag, and all sampled batch row identities. It deliberately does not assign a specific recurrent/batched merge to a particular row when that mapping is not exact.
