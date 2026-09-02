# V3 CLS two-arm evaluator protocol

`src/recursive_geodesic_composition/evaluate_v3_cls.py` is an evaluator-only implementation. It does not import a dataset loader or a teacher and cannot prepare caches or train. No development or official data was accessed while implementing or testing it; tests use generated tensors only.

## Arms

1. **Safe direct arm:** the complete deterministic 1,025–4,096-token prefix is represented by exact contiguous leaves of at most 1,024 tokens. The first leaves are exactly 1,024 tokens and only the final leaf may be short. Two to four children enter one direct K=4 merge. The target is the pinned GTE ModernBERT normalized CLS embedding of that exact prefix. Pooled and per-corpus cosine, fixed logistic/RBF attacks, 400-replicate bootstrap intervals, finite/rank/collapse diagnostics, and merge telemetry are reported.
2. **Full-document arm:** every available token is represented by the same exact contiguous 1,024-token leaves, retaining the final short leaf. Deterministic contiguous K=4 recursion carries singleton groups unchanged and has no token, leaf, or depth cap. Shards carrying any full-root teacher target are rejected. This arm reports only pooled/per-corpus health and recursion telemetry—never teacher cosine, teacher-root metrics, or attacks.

Both arms must contain identical source IDs in identical order. Safe rows and full shards are checked for exact token accounting, finite tensors, frozen V3 ranges, no truncation, and no caps.

## Fail-closed invocation

The immutable protocol is `configs/v3_cls/evaluator_two_arm.yaml`. A later data-stage owner must create a YAML specification containing:

- `v3_cls_experiment: true`, `role` (`development` or `official`), and frozen `seed`;
- `checkpoint` and `checkpoint_sha256`;
- `training_cache_manifest` and `training_cache_manifest_sha256`;
- `cache_manifest` and `cache_manifest_sha256`;
- a create-only `output_dir` under a `v3_cls` namespace.

Run only after the applicable data-access authorization:

```bash
PYTHONPATH=src python scripts/v3_cls/evaluate_two_arm.py --spec <frozen-spec.yaml> --device cuda
```

The checkpoint must be a fresh V3 CLS strict-K4 autoencoder at its fixed final TRAIN-only step. Its config and TRAIN cache manifest are independently checked for V3 namespace, teacher identity/pooling, range, seed, fixed-final selection, and forbidden historical initialization. The evaluation cache requires an atomic `COMPLETE` marker and hashes every safe tensor, full index, and per-document full shard. Output is create-only and records checkpoint, cache, protocol, attack-config, specification, and evaluator hashes.

### Cache schema

The cache manifest declares schema `v3_cls_two_arm_cache_v1`, study `v3_cls`, role and exact allocated source range, pinned CLS teacher provenance, exact `safe_arm`/`full_arm` mappings copied from the protocol, and an `artifacts` mapping whose values are SHA-256 strings or `{sha256: ...}` objects. Each corpus has `<corpus>/safe.pt` and `<corpus>/full_index.json`; each full-index item points to a separately hashed shard. Full shards may contain only leaf/token metadata and must not contain teacher-root fields.
