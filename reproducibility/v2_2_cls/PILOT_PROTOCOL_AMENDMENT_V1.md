# V2.2 pilot amendment V1 — BASELINE reference

This pre-TRAIN amendment preserves `pilot_frozen.yaml` and its three-arm freeze as historical provenance. It adds a fourth 250-step arm, `BASELINE`, solely as the reference required by the original phrase “corresponding baseline pilot.” BASELINE is tensor-identical to A/B/C in seed 424242, fresh initialization, cache rows, row order, minibatches, batch size 8, optimizer/schedule, architecture, cosine/mean/diagonal-variance/existing regularizers; it has no covariance, whitening, corpus-balanced covariance, or residual-mean term.

A/B/C are eligible only after all health gates and final fixed-row cosine >= BASELINE - .002. The lexicographic winner is then lowest frozen covariance/correlation mismatch, residual mean norm, diagonal variance mismatch. BASELINE cannot win. No eligible A/B/C is an explicit rejection and prohibits full V2.2 training.
