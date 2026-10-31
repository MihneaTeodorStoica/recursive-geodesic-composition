# CLEAN DEVELOPMENT ONLY v6 protocol

This is a create-only, outcome-preserving development retraining and validation. It does not authorize access to, freezing of, or execution on protected data. It does not edit the paper.

## Source boundaries

For each of the same four pinned corpora and revisions:

- fresh AE TRAIN scan: `[140000,152000)`;
- fresh development scan: `[152000,161000)`;
- untouched reserve buffer: `[161000,162000)`;
- future protected range: `[162000,192000)`.

The reserve and future ranges are not opened. Dataset loading is lazy and bounded. Cache publication uses a fully fsynced staging directory and atomic rename; all cache/result namespaces are new and create-only.

## Frozen retraining

Seeds are `424242`, `271828`, and `314159`. The genuine strict K4 recipe remains fixed: equal-corpus batch eight, 400 optimizer updates, 300 decoder-only full-arity updates, then 100 joint updates with a nonzero scaled encoder learning rate and the established objective ramp and cosine schedule. The only checkpoint is the fixed final step.

The original encoder-only files were removed for storage. Therefore each seed continues from the encoder state embedded in its current strict-AE checkpoint. Decoder keys from that source are explicitly ignored and the decoder receives its deterministic seeded initialization. This is continued AE retraining, not training from scratch. Before development access, every encoder and decoder parameter tensor must show a nonzero change from its exact initialization.

The TRAIN cache is deterministic and outcome-blind. All three final checkpoints are frozen before the development cache exists. Matched parent-only probes use tensor-identical initialization and an identical deterministic update schedule, are fit only on disjoint groups from fresh v6 TRAIN, and receive per-scope floors from a separate TRAIN-only group set. No development row contributes to fitting or floor establishment.

## Fresh development evaluation

Selection is the first 1000 source-order documents per corpus eligible for a safe fixed direct K4 prefix. Eligibility uses only deterministic length/structure. The primary arm is one direct K4 merge over 1025–4096 tokens with a direct frozen-teacher target.

The same selected documents also receive an unrestricted full-document diagnostic when preparation succeeds efficiently: every available token, exact contiguous leaves, no token/leaf/depth cap, no truncation, and no teacher comparison or AUC. Operational errors fail rather than skip or shorten a selected document.

The fixed grouped standardized logistic and RBF attacks, teacher cosine, built-in parent-only decoder, newly TRAIN-fitted matched probes, output health, collapse checks, and geodesic/importance saturation telemetry are reported for every seed. All outcomes are retained.

## Promotion gate

The sole AUC target is a three-seed mean strongest pooled orientation-safe AUC `<0.60`; each seed also has the frozen `<0.61` guard. Every seed must have pooled teacher cosine `>0.98`, pooled built-in ordered child cosine `>=0.90`, all frozen TRAIN-only retention floors passing, all encoder and decoder parameter tensors changed, finite noncollapsed outputs, and no training/safe/full saturation.

Failure stops at development and does not suppress results. Passing still does not access or freeze protected data; that would require a separate future decision.
