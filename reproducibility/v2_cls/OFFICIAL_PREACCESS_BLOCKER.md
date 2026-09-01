# Official V2 preaccess blocker

**Status: STOP BEFORE OFFICIAL ACCESS.** The V2 official range `[214000,244000)` has not been opened or streamed.

The user-required official evaluator closure cannot truthfully be frozen from the currently implemented V2 artifacts:

1. There are no V2 TRAIN-fitted parent-only probes or frozen probe-floor artifacts. The v1 probe assets are explicitly representation-space-incompatible and cannot be reused.
2. The only V2 development evaluator (`scripts/v2_cls/development_eval.py`) is a deliberately limited safe-span evaluator over 100 documents/corpus. It does not implement the complete official two-arm safe/unrestricted cache, shared official row selection, full-document recursion statistics, probe retention, official bootstrap schema, throughput, or official output manifest required for a one-shot official result.
3. The v1 official-v6 scripts encode v1 namespaces, v1 checkpoint/probe identities, and v1 cache schemas. Repointing them at V2 paths without a reviewed V2-specific implementation would be semantic reinterpretation, not reuse of an established evaluator.
4. No V2 official config/output schema/preaccess validator currently binds the three V2.1 checkpoint hashes, V2 TRAIN-only probes, exact CLS cache provenance, selection rule, bootstrap procedure, and the required safe/full distinction.

Proceeding to the one-shot official range with the limited development script would omit required diagnostics and leave official evaluator semantics ambiguous. That would violate the V2 protocol. This is therefore an explicit scientific-validity preaccess stop, not a negative model result.

Required work before an official freeze:

- fit and freeze V2-only TRAIN parent probes/floors;
- implement and test a V2-specific official cache preparation/evaluator with complete safe and unrestricted arms;
- define/freeze official selection, bootstrap, output schema, artifact hash checks, and failure recovery semantics;
- run synthetic tests without opening official data; and
- commit/push a V2 official preaccess freeze.

No model training, checkpoint selection, or protected-source access is authorized by this blocker.
