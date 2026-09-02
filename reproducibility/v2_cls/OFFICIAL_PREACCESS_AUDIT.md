# V2 official preaccess audit

**Result: FAIL CLOSED / NO ACCESS.**

Completed: V2-only TRAIN probe/floor closure. `TRAIN_PROBE_MANIFEST.json` verifies the three V2.1 checkpoint hashes, V2 CLS TRAIN cache manifest hash, deterministic disjoint fit/floor selections, schedules, probe hashes, and floors. No v1 checkpoint/cache/probe is loaded by `scripts/v2_cls/freeze_train_probes.py`.

Not completed: a complete V2-specific two-arm official cache/evaluator and its synthetic smoke closure. The existing v1 official evaluator cannot be used by path substitution because it encodes v1 cache/probe/checkpoint schemas. The current V2 development evaluator is safe-arm-only and is insufficient for one-shot official access.

Accordingly no machine-readable access freeze is issued and `[214000,244000)` remains unspent. Required tests for a future freeze include synthetic safe/full rows, cache namespace/provenance mismatch rejection, direct-target prohibition on full rows, probe identity/floor checks, selected-row equality across seeds, bootstrap determinism, and no-source-loader preflight imports.

## Updated closure audit

V2 official implementation/config/probe closure has been built and hash-frozen in `OFFICIAL_V2_1_FREEZE.json`. Corpus-free smoke checks passed: exact 1024 chunk coverage including remainder, K4 depth/singleton arithmetic, V2 checkpoint CLS provenance, V2 TRAIN-probe/checkpoint hashes, and refusal of an existing cache/output namespace. No `prepare` invocation occurred. The one-shot command is now explicitly bound in the freeze manifest. Official range remains unopened at this point.
