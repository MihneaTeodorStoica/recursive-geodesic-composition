# V2 CLS interim final report (fail-closed stop before development/official)

## Scientific usability

The corrected CLS extraction implementation and predeclared V2 protocol are scientifically usable as a starting point, but a complete V2 experimental result is **not yet scientifically usable** because training, development evaluation, probes, downstream evaluation, and official evaluation have not been run. The run stopped before any development/official access because the repository does not contain enough information to reproduce the v1 pre-final lineage that produced the continuation source checkpoints.

## What changed from v1

- `FrozenTeacher` now has explicit `pooling` with two paths:
  - `masked_mean` (legacy default) preserves v1 behavior for existing code/configs;
  - `cls` implements documented GTE ModernBERT usage: add special tokens, run the teacher, take `outputs.last_hidden_state[:, 0]`, float32 normalize.
- K-ary/base cache rows and sidecars record teacher ID, revision, tokenizer/model class, and pooling.
- Training refuses declared teacher/pooling mismatches in caches.
- V2 configs live under `configs/v2_cls/` and explicitly require `teacher_pooling: cls` and fresh random initialization.
- V2 protocol/range guards live under `scripts/v2_cls/` and `reproducibility/v2_cls/`.

No v1 artifact, checkpoint, official-v6 result, release asset, or existing downstream diagnostic was intentionally modified.

## Teacher validation

`/venv/main/bin/python scripts/v2_cls/validate_cls_reference.py --device cuda` passed before training. Recorded artifact: `reproducibility/v2_cls/CLS_REFERENCE_VALIDATION.json`.

- Teacher: `Alibaba-NLP/gte-modernbert-base`
- Revision: `e7f32e3c00f91d699e8c43b53106206bcc72bb22`
- Pooling: `cls`
- Max absolute difference against direct model-card-style reference: `0.0`
- Cosines: all reported as `>= 1.0` within float display.

## V1 lineage inspection result

The repository records that clean-v6 final checkpoints continue from source checkpoints such as `results_kary4_strict_autoencoder_seed424_dev_v2/autoencoder_seed424242.pt`. It does not include the exact earlier multi-stage training configs/ranges required to regenerate those source checkpoints from scratch, nor proof that their learned merger/decoder parameters were never trained against masked-mean teacher representations. Therefore V2 must not initialize from those learned v1 checkpoints. The V2 plan predeclares a distinct fresh-random 400-step training protocol rather than pretending to be an exact v1 rerun.

## Declared V2 ranges

Per pinned corpus, guarded by `scripts/v2_cls/protocol.py`:

- prior spent/refused: official-v5 `[110000,140000)`, clean-v6 train/development/buffer `[140000,162000)`, official-v6 `[162000,192000)`;
- V2 TRAIN `[192000,204000)`;
- V2 development `[204000,213000)`;
- V2 buffer `[213000,214000)`;
- future V2 official `[214000,244000)`.

Programmatic non-overlap audit passed in `reproducibility/v2_cls/PROTOCOL_AUDIT.json`.

## Training/evaluation metrics

Not run. No V2 development or official metrics exist. No V2 checkpoints or probes exist. No downstream V2 results exist.

## Validation results

See `reproducibility/v2_cls/VALIDATION.md`.

- Protocol audit: PASS.
- Smoke invariants: PASS (`reproducibility/v2_cls/SMOKE_INVARIANTS.json`).
- Compileall: PASS.
- Pytest subset: `20 passed in 5.92s`.

## Remaining concerns

1. Exact v1 pre-final lineage is missing from this publication repository.
2. The proposed fresh V2 protocol is not an exact rerun of v1 and should be reviewed/accepted before expensive training.
3. Development/official evaluators and downstream V2 scripts still need to be implemented/frozen before accessing V2 development/official ranges.
4. Official data must remain unopened until models, checkpoint selection, attacks, probes, evaluator code, and downstream protocol are frozen and hash-recorded.

## Artifact hashes

Implementation/provenance hashes are recorded in `reproducibility/v2_cls/V2_IMPLEMENTATION_SHA256SUMS`.

## Future release/PR contents proposed

A future `v2-paper` release/PR should include, after successful training/audit only:

- V2 CLS source code/config/protocol files;
- three V2 step-400 fresh checkpoints and SHA-256 hashes;
- V2 TRAIN/development/official cache manifests (not necessarily raw caches if too large);
- V2-fitted parent-only probes and hashes;
- machine-readable development, official, and downstream results;
- V2 figures generated only from V2 result files;
- final audit proving no v1 artifacts changed and official access happened after freeze.
