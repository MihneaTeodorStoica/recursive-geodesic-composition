# V4 Euclidean run log (append-only)

## Initialization
- Created `v4-paper-euclidean` from frozen audited V3 commit `0a1146cedad05ff3c75d8c9ed723993ae88212cc` solely to reuse architecture-independent code/provenance fixes. No V3 learned state is permitted.
- Froze fresh per-corpus source roles after all V3 reserved ranges: TRAIN `[348000,360000)`, development `[360000,369000)`, buffer `[369000,370000)`, official `[370000,400000)`.
- Independently validated documented normalized CLS before V4 source access: max_abs_diff=0.0.
- Implemented exactly parameter-matched geodesic, Euclidean-residual, and free-Euclidean geometry switches; each full model has 123,616,258 parameters. Synthetic K=2/3/4 outputs are finite/unit-normalized and tests pass.
