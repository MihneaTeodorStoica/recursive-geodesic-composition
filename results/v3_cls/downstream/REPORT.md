# V3 CLS downstream retrieval report

## Protocol and data

- The blind length inspection and immutable protocol were written before model results: `BLIND_LENGTH_INSPECTION.json` and `PRECOMMIT.json` (SHA-256 `a5aec09c0199b286d7c702915f345b03b32f0f0109db4fde126c11b87ec392df`).
- MLDR English was inspected from pinned revision `d67138e705d963e346253a80e59676ddb418810a`. Its 200,000 documents and 984,905,843-byte compressed corpus were infeasible for the available 5.7 GB workspace and exact float32 leaf-cache protocol.
- The precommitted long-document benchmark is LongEmbed NarrativeQA, pinned to immutable revision `10039a580487dacecf79db69166e17ace3ede392`: 355 documents, 10,449 queries, and 10,449 qrels. The corpus contains 30,413,806 teacher tokens, 29,882 exact 1,024-token leaves, depths up to 5, and a maximum document length of 488,429 tokens.
- NFCorpus uses the standard BEIR test qrels: 3,633 documents, 323 test queries, and 12,334 judgments. `PRECOMMIT_AMENDMENT_1.json` was frozen before model construction after the fail-closed loader established that the common `queries.jsonl` has all dataset splits (3,237 rows) and the test qrels identify 323. `SOURCE_SELECTION_AUDIT.json` records the explicit 2,914 non-test exclusions; no test query, qrel, or corpus document was dropped.
- All methods use the pinned normalized-CLS teacher and one hash-addressed float32 leaf cache per benchmark. Recursive roots for all three final checkpoints and both leaf-mean baselines consume those identical tensors. Direct teacher is reported for NFCorpus because every query/document fits 8,190 raw tokens; it is omitted for NarrativeQA rather than truncating.

## Results

Metrics are on a 0–1 scale.

### NFCorpus test

| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| recursive seed 424242 | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| recursive seed 271828 | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| recursive seed 314159 | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| token-weighted leaf mean | 0.348162 | 0.168303 | 0.317419 | 0.550243 |
| unweighted leaf mean | 0.347681 | 0.168243 | 0.317306 | 0.549805 |
| direct teacher | 0.348292 | 0.169335 | 0.317520 | 0.549005 |

NFCorpus has only nine multi-leaf documents (maximum three leaves), which explains the equal rounded recursive metrics.

### LongEmbed NarrativeQA

| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| recursive seed 424242 | 0.549882 | 0.666475 | 0.890612 | 0.513091 |
| recursive seed 271828 | 0.551776 | 0.670495 | 0.892238 | 0.514432 |
| recursive seed 314159 | 0.548926 | 0.666380 | 0.891760 | 0.511948 |
| token-weighted leaf mean | 0.560881 | 0.674419 | 0.886783 | 0.525044 |
| unweighted leaf mean | 0.560153 | 0.672792 | 0.885252 | 0.524565 |

## Artifacts and runtime

- Aggregate: `SUMMARY.json`
- Full metrics, token/leaf/depth strata, provenance, and artifact hashes: `<dataset>/RESULTS.json`
- Per-query metrics: `<dataset>/<method>_per_query.jsonl`
- Top-100 TREC runs: `<dataset>/rankings/<method>.trec`
- Document embeddings: `<dataset>/<method>_document_embeddings.npz`
- Exact leaves and tensor hashes: `<dataset>/leaf_cache/MANIFEST.json` and `documents/*.pt`
- Logs: `PRECOMMIT_RUNTIME.log`, `RUN_RUNTIME.log`

Measured cache preparation was 164.9 s (NFCorpus) plus 1,141.5 s (NarrativeQA). The successful final evaluation pass was 152.5 s wall time (22.9 s NFCorpus, 119.5 s NarrativeQA inside dataset evaluators). End-to-end elapsed compute including the intentional fail-closed/retry checks was approximately 25 minutes. Downstream artifacts occupy about 635 MB.
