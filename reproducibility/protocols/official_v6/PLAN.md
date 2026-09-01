# One-shot official strict-K4 AE v6 protocol

## State and prohibition

This directory is a **preaccess freeze only** based on committed clean-v6 development `f38f75d2d70a9af92d3bdeae959281173c7998ee`, which passed before this protocol was written. Preparation and evaluation have not been launched. No paper file is part of this change.

The only authorized corpus scan is `[162000,192000)` for each pinned corpus. The stream guard refuses every other interval before importing the corpus loader. The declared contiguous refusal partition covers `[0,162000)`, explicitly including official v5, clean-v6 fresh TRAIN, clean-v6 fresh development, and `[161000,162000)` as a mandatory buffer.

## Frozen population

For each of the same four pinned corpora, scan in source order and accept the first 1000 nonblank documents whose deterministic safe prefix contains 1025--4096 teacher tokens. Eligibility uses no model output. Once eligible, a document cannot be skipped, shortened, or replaced after an operational failure. Both arms use the identical accepted source IDs and order.

## Mandatory arms

1. **Safe direct arm.** Exact contiguous leaves of at most 1024 tokens cover the 1025--4096-token prefix. A single direct strict-K4 merge (two, three, or four active children) produces the generated root. The direct frozen-teacher embedding of that same prefix is its target.
2. **Unrestricted full-document arm.** Every available token is covered by exact contiguous leaves and deterministic K4 recursion with singleton carry. There is no token cap, leaf cap, depth cap, or truncation. Storage or inference failure is fatal. This arm has no teacher target, AUC, teacher cosine, or attack.

## Frozen models and probes

The three exact local clean-v6 checkpoints and SHA-256 hashes are in `config.yaml`. Before any corpus/cache access, verification also checks each exact source checkpoint, per-seed pre-development freeze, actual autoencoder variant, strict decoder state, finite tensors, fixed final step, and genuine nonzero change in every encoder and decoder parameter tensor.

Each candidate uses its own newly clean-v6-TRAIN-fitted parent-only probe. Candidate and source probe hashes and metadata are checked, with no official fitting or update. Every pooled and per-corpus candidate score is compared with the corresponding TRAIN-only floor frozen before development.

## Frozen analysis

For every seed and pooled/per-corpus scope, report teacher cosine and fixed grouped standardized logistic and RBF attacks. Source-document groups cannot cross train/holdout. Every confidence interval uses 2000 source-group bootstrap replicates. Three-seed strongest-attack points and shared-heldout-group intervals are reported pooled and per corpus.

The built-in strict decoder receives generated parents only. On exact-four safe rows, report ordered reconstruction cosine and global ordered-child retrieval (recall and reciprocal rank), pooled and per corpus. On the full arm, decode every internal merge and report active-child reconstruction by depth, child type, corpus, and their combinations.

Report rank, finite fraction, rounded uniqueness, collapse, norm and saturation health; geometry/importance saturation; token/leaf/depth distributions; exact merge counts; and preparation/inference throughput. Any nonfinite value or accounting mismatch is fatal.

## Reporting-only gate

The sole AUC target is strict `<0.60` for the three-seed mean strongest pooled point. The per-seed guard is derived as that sole target plus the frozen margin in `config.yaml`. Also require every seed's pooled teacher cosine to exceed the frozen cosine bound, every seed/safe scope built-in exact-four child cosine to meet its floor, every seed/scope frozen TRAIN-probe floor to pass, no collapse/nonfinite/saturation, and complete unrestricted processing.

All outcomes are published atomically regardless of gate result. Post-access tuning, selection, rerun, suppression, and paper editing are forbidden.

## One-shot publication

`run.sh` requires a clean tracked tree at the annotated preaccess tag, verifies `PREDECLARATION_SHA256SUMS` and all immutable inputs, and refuses any existing unique marker/cache/result/staging namespace. Preparation and evaluation claim separate exclusive markers and publish only by atomic rename. All shards, indexes, manifests, results, and reports carry hash guards; free-space guards fail instead of truncating.
