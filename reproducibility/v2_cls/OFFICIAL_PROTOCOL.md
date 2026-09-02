# V2.1 official protocol draft (not yet an access authorization)

## Frozen components completed in this phase

- Teacher: `Alibaba-NLP/gte-modernbert-base` at `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, pooling `cls`.
- Candidate checkpoints and SHA-256 identities are in `TRAIN_PROBE_MANIFEST.json`.
- V2 TRAIN-only parent-only probes were fit on 80 deterministic exact-K4 groups/corpus and floors on a disjoint 80 groups/corpus. The candidate-only V2 role is used because no same-space source model exists; v1 source probes are explicitly forbidden.
- Fit: AdamW, seed 730241 initialization, seed 730242 fixed 400x128 schedule, LR .0003, weight decay .001, cosine plus .25 MSE objective, floor = held-out V2 TRAIN score - .001.

## Required official semantics

Official range remains `[214000,244000)` and unopened. Safe rows must be selected source-order, same rows all seeds, direct CLS target only on complete 1025--4096-token prefix; exact K=4 chunking. Full arm must recursively embed every token in each selected document with 1024-token leaves and no full-root target/AUC/cosine. Logistic/RBF attacks use `configs/evaluator_attacks_v1.yaml`, orientation-safe semantics, grouped bootstrap 400 replicates.

## Not an authorization

The V2 official cache preparer/evaluator implementing all of these semantics, full-document statistics, strict reconstruction, probe application, and a create-only one-shot output schema is not complete. Therefore this document does not authorize official access.

## V2.1 frozen implementation

`configs/v2_cls/official_v2_1.yaml` and `scripts/v2_cls/official_v2_1.py` implement create-only preparation/evaluation. Preparation is the sole corpus entrypoint; import, preflight, chunk/depth tests, and checkpoint/probe identity checks do not import `datasets` or call the official range. Preparation claims a one-shot spent marker before streaming, selects the first 1000 source-order documents per corpus with complete 1025--4096 safe prefix, derives both arms from each accepted document, and atomically publishes the cache. Evaluation reports safe paired CLS cosine/attacks/per-corpus decoder/probe results and full recursion-only records without full-root teacher targets.
