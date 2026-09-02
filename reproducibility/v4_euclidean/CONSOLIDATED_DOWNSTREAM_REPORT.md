# Consolidated frozen V4 downstream benchmark report

## Protocol and provenance

No V4 model was trained, selected, or altered for this suite. All recursive results use the three frozen Euclidean-residual checkpoints. The teacher is normalized documented `last_hidden_state[:,0]` from `Alibaba-NLP/gte-modernbert-base@e7f32e3c00f91d699e8c43b53106206bcc72bb22`.

LongEmbed is pinned to `dwzhu/LongEmbed@10039a580487dacecf79db69166e17ace3ede392`. Every benchmark file was verified content-identical to MTEB metadata revision `6e346642246bfb4928c560ee08640dc84d074e8c`. The official `corpus`, `queries`, and `qrels` configurations are evaluated as the test set. NFCorpus uses the BEIR test split and archive SHA-256 `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`.

Each task uses one hash-checked CLS leaf cache shared tensor-identically by all recursive seeds and both pooling baselines. Queries use direct CLS. Metrics are BEIR-compatible `pytrec_eval` nDCG/recall plus reciprocal rank over the same deterministic rankings. Query bootstrap confidence intervals use 1,000 replicates. Direct CLS appears only when a complete candidate corpus/condition fits without truncation.

## Dataset scale

| task | documents | queries | total tokens | total leaves | merges | max depth | token median / p90 / max | leaf median / p90 / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nfcorpus | 3633 | 323 | 1203147 | 3642 | 8 | 1 | 331 / 465 / 2232 | 1 / 1 / 3 |
| narrativeqa | 355 | 10449 | 30413806 | 29882 | 10197 | 5 | 66219 / 158134 / 488429 | 65 / 155 / 477 |
| qmsum | 197 | 1527 | 2748767 | 2781 | 1004 | 3 | 12783 / 23924 / 37555 | 13 / 24 / 37 |
| 2wikimqa | 300 | 300 | 2682849 | 2768 | 997 | 3 | 7924 / 15667 / 16725 | 8 / 16 / 17 |
| summ_screen_fd | 336 | 336 | 3001115 | 3091 | 1151 | 3 | 8482 / 14030 / 24688 | 9 / 14 / 25 |

Per-document token count, leaf count, recursion depth, merge count, and synthetic context condition are preserved in `results/v4_euclidean/publication_sources/document_scale_metadata.csv`.

## Real-world retrieval

All values are on 0–1 scale. “V4 mean” is the arithmetic mean of the three frozen-seed metrics.

| task | method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---:|---:|---:|---:|
| nfcorpus | V4 seed 424242 | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| nfcorpus | V4 seed 271828 | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| nfcorpus | V4 seed 314159 | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| nfcorpus | V4 mean | 0.348170 | 0.168303 | 0.317527 | 0.550243 |
| nfcorpus | token-weighted | 0.348162 | 0.168303 | 0.317419 | 0.550243 |
| nfcorpus | unweighted | 0.347681 | 0.168243 | 0.317306 | 0.549805 |
| nfcorpus | direct CLS | 0.348292 | 0.169335 | 0.317520 | 0.549005 |
| narrativeqa | V4 seed 424242 | 0.551721 | 0.668868 | 0.890612 | 0.514795 |
| narrativeqa | V4 seed 271828 | 0.545648 | 0.664370 | 0.889750 | 0.508257 |
| narrativeqa | V4 seed 314159 | 0.551488 | 0.668198 | 0.890899 | 0.514709 |
| narrativeqa | V4 mean | 0.549619 | 0.667145 | 0.890420 | 0.512587 |
| narrativeqa | token-weighted | 0.560881 | 0.674419 | 0.886783 | 0.525044 |
| narrativeqa | unweighted | 0.560153 | 0.672792 | 0.885252 | 0.524565 |
| qmsum | V4 seed 424242 | 0.464565 | 0.653569 | 0.978389 | 0.405474 |
| qmsum | V4 seed 271828 | 0.461557 | 0.651604 | 0.979699 | 0.402049 |
| qmsum | V4 seed 314159 | 0.463780 | 0.652914 | 0.981009 | 0.404540 |
| qmsum | V4 mean | 0.463301 | 0.652696 | 0.979699 | 0.404021 |
| qmsum | token-weighted | 0.481271 | 0.680419 | 0.984283 | 0.419354 |
| qmsum | unweighted | 0.475386 | 0.667322 | 0.983628 | 0.415590 |
| 2wikimqa | V4 seed 424242 | 0.830818 | 0.906667 | 0.986667 | 0.807179 |
| 2wikimqa | V4 seed 271828 | 0.827452 | 0.903333 | 0.986667 | 0.803647 |
| 2wikimqa | V4 seed 314159 | 0.833201 | 0.906667 | 0.986667 | 0.810110 |
| 2wikimqa | V4 mean | 0.830490 | 0.905556 | 0.986667 | 0.806978 |
| 2wikimqa | token-weighted | 0.818614 | 0.903333 | 0.983333 | 0.791688 |
| 2wikimqa | unweighted | 0.820370 | 0.893333 | 0.983333 | 0.796758 |
| summ_screen_fd | V4 seed 424242 | 0.970515 | 0.997024 | 1.000000 | 0.961607 |
| summ_screen_fd | V4 seed 271828 | 0.971986 | 0.997024 | 1.000000 | 0.963690 |
| summ_screen_fd | V4 seed 314159 | 0.972506 | 0.997024 | 1.000000 | 0.964335 |
| summ_screen_fd | V4 mean | 0.971669 | 0.997024 | 1.000000 | 0.963211 |
| summ_screen_fd | token-weighted | 0.967989 | 0.997024 | 1.000000 | 0.958333 |
| summ_screen_fd | unweighted | 0.969477 | 0.997024 | 1.000000 | 0.960424 |

V4-mean nDCG@10 query-bootstrap intervals and paired pooling gaps:

| task | V4 nDCG@10 95% CI | V4 − token-weighted (95% paired CI) | V4 − unweighted (95% paired CI) |
|---|---:|---:|---:|
| nfcorpus | [0.313758, 0.381560] | 0.000008 [-0.000000, 0.000024] | 0.000489 [0.000098, 0.001065] |
| narrativeqa | [0.541347, 0.557798] | -0.011262 [-0.013882, -0.008508] | -0.010534 [-0.013007, -0.007830] |
| qmsum | [0.443067, 0.483497] | -0.017970 [-0.027823, -0.007634] | -0.012085 [-0.022965, -0.001710] |
| 2wikimqa | [0.789413, 0.867631] | 0.011876 [-0.006707, 0.032564] | 0.010120 [-0.011816, 0.033696] |
| summ_screen_fd | [0.959679, 0.984008] | 0.003679 [-0.005273, 0.012389] | 0.002192 [-0.006085, 0.011290] |

Direct CLS is present for NFCorpus. It is excluded from every real LongEmbed task because at least one corpus document exceeds 8,190 raw teacher tokens; no truncated subset was substituted.

## Scale-stratified result

Bins were frozen before model evaluation. Detailed token/leaf/depth values, all requested metrics, query counts, and paired confidence intervals are in `CONSOLIDATED_SUITE_RESULTS.json` and the publication CSVs. The clearest nDCG@10 comparisons against token-weighted pooling are:

- NarrativeQA: +0.0574 at 4,097–8,190 tokens, +0.0367 at 8,191–16,384, then −0.0154 at 32,769–65,536 and −0.0135 above 65,536. At depth 5 (318 queries), the gap is −0.0355, 95% CI [−0.0504, −0.0215].
- QMSum: −0.0179 at 8,191–16,384 and −0.0255 at 16,385–32,768. At depth 3 (507 queries), the gap is −0.0253 [−0.0420, −0.0108].
- 2WikiMultihopQA: +0.0182 at 4,097–8,190 and +0.0129 at 8,191–16,384, with confidence intervals spanning zero.
- SummScreenFD: gaps are small and non-monotonic; the overall +0.00368 and all substantive strata intervals include zero.

Thus there is no increasing recursive advantage with token count, leaf count, or depth. On the two datasets with substantial genuinely deep populations, the longest/deepest strata favor simple pooling.

## LongEmbed Needle and Passkey

The standard eight context conditions are evaluated independently with 100 candidate documents and 50 queries each. The official synthetic main score is nDCG@1. Direct CLS is shown only through context 8,192, where every actual tokenized document fits.

| task | context | V4 mean | token-weighted | unweighted | direct CLS |
|---|---:|---:|---:|---:|---:|
| needle | 256 | 0.880000 | 0.880000 | 0.880000 | 0.880000 |
| needle | 512 | 0.540000 | 0.540000 | 0.540000 | 0.540000 |
| needle | 1024 | 0.220000 | 0.220000 | 0.220000 | 0.220000 |
| needle | 2048 | 0.226667 | 0.160000 | 0.140000 | 0.160000 |
| needle | 4096 | 0.146667 | 0.220000 | 0.180000 | 0.060000 |
| needle | 8192 | 0.120000 | 0.140000 | 0.120000 | 0.060000 |
| needle | 16384 | 0.026667 | 0.040000 | 0.040000 | — |
| needle | 32768 | 0.040000 | 0.120000 | 0.120000 | — |
| passkey | 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| passkey | 512 | 0.980000 | 0.980000 | 0.980000 | 0.980000 |
| passkey | 1024 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| passkey | 2048 | 0.420000 | 0.800000 | 0.820000 | 0.960000 |
| passkey | 4096 | 0.280000 | 0.560000 | 0.540000 | 0.640000 |
| passkey | 8192 | 0.080000 | 0.200000 | 0.200000 | 0.360000 |
| passkey | 16384 | 0.086667 | 0.140000 | 0.160000 | — |
| passkey | 32768 | 0.020000 | 0.060000 | 0.040000 | — |

Across-context nDCG@1 means: Needle V4 0.2750 versus token-weighted 0.2900 and unweighted 0.2800; Passkey V4 0.4833 versus 0.5925 and 0.5925. Performance falls sharply with context length for every method. Recursive composition does not improve that scaling and is especially worse on Passkey.

## ECtHR-A multilabel classification

Frozen representations use official train/validation/test separation, one-vs-rest logistic regression, a predeclared probability threshold of 0.5, validation micro-F1 selection over the frozen C grid with smaller-C tie breaking, refit on train+validation, and one test evaluation.

| method | micro-F1 | macro-F1 |
|---|---:|---:|
| V4 seed 271828 | 0.643898 | 0.518392 |
| V4 seed 314159 | 0.646844 | 0.541833 |
| V4 seed 424242 | 0.641096 | 0.533861 |
| token_weighted_mean | 0.635902 | 0.505216 |
| unweighted_mean | 0.634891 | 0.515470 |
| **V4 mean** | **0.643946** | **0.531362** |

V4 exceeds both pooling baselines on this frozen linear-probe task (mean micro-F1 0.643946; macro-F1 0.531362).

## 20 Newsgroups clustering

All representations use identical fixed K-means seeds/settings.

| method | NMI | ARI |
|---|---:|---:|
| V4 seed 271828 | 0.595915 | 0.458106 |
| V4 seed 314159 | 0.612930 | 0.463768 |
| V4 seed 424242 | 0.613061 | 0.463918 |
| token_weighted_mean | 0.597390 | 0.463594 |
| unweighted_mean | 0.614374 | 0.472241 |
| **V4 mean** | **0.607302** | **0.461931** |

V4 is approximately tied with token-weighted pooling and below unweighted pooling.

## Failures, exclusions, and audit notes

- New-task schema/ID/qrel inspection found no malformed, duplicate, missing, or qrel-referenced absent examples. Metric evaluation dropped no queries.
- The first QMSum recursive pass encountered a telemetry API invocation error before any metric or result was computed. Engineering amendment 1 changed only positional arguments to required keyword arguments; caches, embeddings, rankings, models, bins, and metrics were unchanged.
- Needle and Passkey were computationally feasible; neither was excluded or altered.
- Direct teacher exclusions are solely context-window validity exclusions described above.
- ECtHR-A and 20 Newsgroups were already run under the frozen V4 downstream protocol and were not retuned.

## Factual conclusion

V4 does **not materially outperform simple pooling overall**, and it does not gain relative advantage on genuinely long/deep documents. It modestly improves 2WikiMultihopQA and SummScreenFD, loses on NarrativeQA and QMSum, loses the synthetic context-scaling comparison—substantially on Passkey—and is worse in the deepest NarrativeQA/QMSum strata. Its clearest downstream gain is ECtHR-A linear classification, while clustering is no better than pooling.

All machine-readable results, confidence intervals, document metadata, figure sources, and figures are under `results/v4_euclidean/` and `figures/v4_euclidean/downstream_suite/`.
