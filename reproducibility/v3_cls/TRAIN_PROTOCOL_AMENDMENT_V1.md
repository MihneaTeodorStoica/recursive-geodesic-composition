# V3 TRAIN protocol amendment V1 — fresh merger learning rate and staged joint run

Status: predeclared after full TRAIN cache construction and before any development access.

## TRAIN-only diagnosis
The initial 20-step tiny engineering pilot inherited V1's continuation-stage `encoder_learning_rate_scale: 0.02`. Its merger LR peaked at only `4e-7`; this setting assumed a useful learned merger and is not technically justified for V3 fresh-from-random training. Inspection of V2 code/configs was used only as a hypothesis source and confirmed that its fresh merger stage used scale 1.0 and LR `1e-4`; no V2 state or data artifact is used.

## Bounded diagnostic
Run exactly one 80-step full-architecture TRAIN-only diagnostic with seed 424242, batch 8, LR `1e-4`, scale 1.0, 10-step warmup, cosine floor `1e-5`, and merger/distribution objectives only. Selection criterion: all values finite, no sustained geodesic saturation, no exploding gradient (`grad_norm < 100`), and final 20-step mean paired-teacher loss lower than first 20-step mean. This run is diagnostic and cannot become a final checkpoint.

## Candidate final protocol if diagnostic passes
One uninterrupted fresh-random 2400-step autoencoder run per seed, K=4 V1 final architecture, batch 8, AdamW, LR `1e-4`, encoder scale 1.0, warmup 160, cosine floor `1e-5`, weight decay 0.001. Steps 1–1200 meaningfully train merger with paired cosine, corpus-balanced mean/variance distribution terms and recursive targets. Strict parent-only decoder/reconstruction/cycle/importance regularizers begin at step 1201 and ramp linearly to inherited weights over 400 steps; all objectives then train jointly through step 2400. Fixed final-step checkpoint; no validation-dependent checkpoint selection.

Inherited coefficients remain: mean 0.25, variance 256, reconstruction 0.15, cycle 5.0, importance 0.025, importance prior/delta/sharpness/slot 0.01 each. Final seeds are fixed at 424242, 271828, 314159 and use identical configs except seed/output path.

If the bounded diagnostic fails its mechanical criteria, preserve it and write a new amendment before another TRAIN-only candidate. No development or official rows may be accessed during this decision.
