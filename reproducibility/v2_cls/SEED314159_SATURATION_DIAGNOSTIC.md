# Seed 314159 Stage-A saturation diagnostic

## Scope and preservation

This is a post-run diagnostic over the already-created V2 CLS TRAIN cache, raw Stage-A logs, frozen configs, and existing successful-seed checkpoints/logs. It did **not** load V2 development, buffer, official, or downstream data; it did not resume or recreate seed `314159`; it did not change any training config or checkpoint. The failed Stage-A directory remains exactly as produced.

Supporting machine-readable records:

- `SEED314159_STEP276_BATCH.json`: deterministic batch identity and K=4 merge accounting;
- `SEED314159_COMPARISON_STEPS260_276.json`: logged TRAIN telemetry for all seeds around this step;
- `SEED314159_SATURATION_DIAGNOSTIC.json`: compact derived facts and source hashes.

## Exact meaning of the rejection

The failure happened at **Stage-A optimizer step 276**. The trainer collects the angle from every genuine `model.merge` call in the optimizer batch (`model._geodesic_angles`), then records

```python
geodesic_saturation_fraction = (angles >= .95 * model.max_angle).float().mean()
```

For this run, `max_angle=0.5`, therefore the diagnostic threshold is **0.475 radians**. The recorded fraction was `0.1111111119389534`, which is float32 `1/9`: exactly **one of nine genuine K=4 merges** in the eight-row batch had bounded correction angle at or above 0.475 rad. It is not 11% of documents, tokens, slots, or training steps.

The trainer then increments the consecutive-step saturation streak whenever that fraction is nonzero. The frozen V2 config set `reject_on_geodesic_saturation: true`, `geodesic_saturation_patience_steps: 1`, and `continue_after_quality_rejection: false`. Hence a single qualifying merge on this one optimizer step produced:

```
geodesic_saturation_nonzero_for_1_steps:0.111111112
```

and terminated the run after logging the completed step. No selected checkpoint existed because final-only checkpoint selection was at step 1200.

## Step-276 batch and denominator

The deterministic Python sampling stream is recoverable without model replay: with `random.seed(314159)`, equal stratified corpus sampling, and the immutable TRAIN cache, the step-276 batch contains two rows per corpus.

| Corpus | Cache row | Leaves | Content tokens | Genuine merges |
|---|---:|---:|---:|---:|
| FineWeb-Edu | 4503 | 2 | 1133 | 1 |
| Wikipedia | 2605 | 5 | 3917 | 2 |
| FineWeb | 3096 | 2 | 1230 | 1 |
| OpenWebText | 3075 | 2 | 1066 | 1 |
| Wikipedia | 5367 | 2 | 1064 | 1 |
| OpenWebText | 1152 | 2 | 1690 | 1 |
| FineWeb | 2973 | 2 | 1660 | 1 |
| FineWeb-Edu | 3347 | 2 | 1384 | 1 |

The total is `1+2+1+1+1+1+1+1 = 9`. The raw source identifiers and level/group layout are retained in `SEED314159_STEP276_BATCH.json`.

The existing telemetry identifies **one** saturated merge but not which of those nine. The trainer does not serialize per-merge angle, raw tangent, or batch IDs into its JSONL record, and the run has no step-275 checkpoint. Identifying the exact row/merge or exact raw tensor values would require replaying optimizer updates, which this diagnostic deliberately does not do.

## Tangent construction and what is known

For `tangent_expmap`, code computes projected `tangent`, then

```python
geodesic_correction = tangent * tanhc(raw_tangent_norm / max_angle)
angle = norm(geodesic_correction)
```

so in exact arithmetic `angle = max_angle * tanh(raw_tangent_norm / max_angle)`. The output stays inside the open 0.5-radian ball and is formed with stable `sinc`/cosine operations.

At step 276 the logged batch values are:

- angle mean `0.0906484351`;
- angle p95 `0.3393569887`;
- angle max / correction-norm max `0.4999998808`;
- saturation count `1/9`;
- threshold `0.475`;
- semantic-correction saturation fraction `0.0`.

The exact saturated raw tangent norm/vector and individual correction vector were not logged and cannot be recovered without prohibited replay. Using the displayed max with the analytic tanh relation gives an **approximate** raw tangent norm of `3.9856`; the raw norm associated with the 0.475 diagnostic threshold is only `0.91589`. These are derived values, not observed tensors. The bounded angle is numerically close to its asymptote but no NaN/Inf occurred.

## Local comparison

Steps 260--275 for seed 314159 had zero geodesic and semantic-correction saturation. At step 275, max angle was `0.05409`, mean `0.04308`, teacher loss `0.02107`, and gradient norm `0.16857`. At step 276 these became `0.50000`, `0.09065`, `0.03530`, and `1.07328` respectively. Distribution mean/variance losses also increased (`0.00731 -> 0.02055`, `1.37e-5 -> 4.70e-5`). This is a real localized adverse batch/update event, not a logging zero/rounding artifact.

At the same optimizer step, the two successful seeds had no saturation:

| Seed | Teacher loss | Grad norm | Angle mean / max | Geodesic saturation | Semantic-correction saturation |
|---:|---:|---:|---:|---:|---:|
| 424242 | 0.01655 | 0.16636 | 0.04165 / 0.05398 | 0 | 0 |
| 271828 | 0.01666 | 0.24318 | 0.03505 / 0.05429 | 0 | 0 |
| 314159 | 0.03530 | 1.07328 | 0.09065 / 0.49999988 | 1/9 | 0 |

The complete recorded comparison is in `SEED314159_COMPARISON_STEPS260_276.json`. Existing final TRAIN-only audits show finite unit-norm, noncollapsed roots for the two completed seeds. The rejected seed has no post-step checkpoint and the logger does not record rank/root norms, so equivalent post-step output/rank claims for seed 314159 cannot be made. The run's finite loss, finite gradient, and successful `clip_grad_norm_(..., error_if_nonfinite=True)` do establish that no non-finite condition was detected at the rejected step.

## Hard-stop rule audit

The strict single-step abort was **not** inherited from clean-v6 final training: `configs/final_v6/*.yaml` set `reject_on_geodesic_saturation: false`. The general K=4/K=8 validation configs did contain `reject_on_geodesic_saturation: true` and a one-step patience, so the concept had published precedent. However, applying it as an immediate terminating Stage-A V2 rule (`continue_after_quality_rejection: false`) was a newly specified V2 choice, not a recovered v1 continuation setting.

A correction near 0.5 does not by itself make the model invalid: tangent-expmap deliberately bounds it, and the threshold is a conservative diagnostic at 95% of the bound rather than a numerical-validity boundary. It does indicate that one raw tangent entered the strongly compressed part of tanh (about 4.35 times the threshold-equivalent raw norm), where gradients through the radial bound are small. Thus it is meaningful health telemetry, but one merge in one nine-merge batch is insufficient evidence of global divergence.

## Conclusion

**Best-supported conclusion: (B) isolated but real saturation that warrants a more appropriate persistence-based safety rule.**

Evidence supports “real” because the recorded maximum is essentially the explicit bound and accompanies a loss/gradient/moment spike. Evidence supports “isolated” because the preceding 16 logged steps had zero saturation, semantic corrections did not saturate, and the event was exactly one merge. It is not supported as (A) divergence: there was no NaN/Inf, no unbounded angle, and no persistence evidence. It is not supported as (C): the numerator/denominator, threshold, and tanh construction are internally consistent. Per-merge tensors are unavailable, so the conclusion remains limited to recorded evidence.

## Proposal only — no amendment made

If separately approved, a **new** V2 protocol should apply identically to every seed and rerun all seeds from fresh random initialization; it must not resume or selectively exempt seed 314159. A defensible amendment is:

- retain the same 0.475-radian per-merge telemetry threshold and log all events;
- set `geodesic_saturation_patience_steps: 3`, rejecting only if any qualifying merge occurs on **three consecutive optimizer steps**;
- retain fail-fast rejection once that persistence condition is met;
- add per-merge batch IDs, raw projected-tangent norm, bounded correction norm/angle, and active arity to the training JSONL before rerunning;
- predeclare these changes and the full all-seed rerun before any training, with no development/official access.

This proposal is based on the bounded geometry and the observed single-step event, not on a favorable seed outcome. It is not active and does not alter the frozen protocol or failed run.
