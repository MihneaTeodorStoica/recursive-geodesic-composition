# V4 Euclidean run log (append-only)

## Initialization
- Created `v4-paper-euclidean` from frozen audited V3 commit `0a1146cedad05ff3c75d8c9ed723993ae88212cc` solely to reuse architecture-independent code/provenance fixes. No V3 learned state is permitted.
- Froze fresh per-corpus source roles after all V3 reserved ranges: TRAIN `[348000,360000)`, development `[360000,369000)`, buffer `[369000,370000)`, official `[370000,400000)`.
- Independently validated documented normalized CLS before V4 source access: max_abs_diff=0.0.
- Implemented exactly parameter-matched geodesic, Euclidean-residual, and free-Euclidean geometry switches; each full model has 123,616,258 parameters. Synthetic K=2/3/4 outputs are finite/unit-normalized and tests pass.
- Built one shared V4 TRAIN CLS cache from frozen `[348000,360000)` (usable rows: 7618/3774/8499/6396 by corpus).
- Ran matched 80-step full-width pilots. All were finite and improving. Free Euclidean revealed an evaluator-only telemetry bug: anchor-output angles were incorrectly compared to the geodesic cap despite the architecture having no cap. Preserved the run, changed Euclidean saturation telemetry to identically zero without changing model/loss, and reran the exact pilot.
- Pilot pass: geodesic teacher loss 0.02970→0.02770; Euclidean residual 0.02987→0.02771; free Euclidean 0.48432→0.36264 (first/last 20 means). Max gradients 0.86/0.93/6.12, all finite and noncollapsed. Froze unchanged 2400-step nine-model comparison protocol and hashes before development.
- Completed all nine fixed 2400-step fresh runs under the identical cache/budget and fitted nine TRAIN-only probes. Geodesic seed 314159 saturated at its 0.5-radian cap and is retained as a protocol-defined negative result; Euclidean arms had no cap and zero cap telemetry. Froze every checkpoint/probe/evaluator/cache/config/attack hash before development access.
