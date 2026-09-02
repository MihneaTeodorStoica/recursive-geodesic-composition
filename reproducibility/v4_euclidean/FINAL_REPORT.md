# V4 Euclidean architecture ablation — final report

## Status and question

V4 is complete. It tested whether V3's detectable recursive signature was primarily caused by spherical/geodesic composition. The answer is **no under this matched experiment**: removing tangent projection, exp-map, interpolation, angular clipping, and spherical-only regularization did not materially lower orientation-safe attack AUC. The selected Euclidean residual model is numerically stable and high-fidelity, but remains distinguishable.

This conclusion is bounded to the tested parameterization, schedule, sources, and teacher. It does not prove that geometry can never matter.

## Frozen design and provenance

- Branch base: finalized/audited V3 commit `0a1146cedad05ff3c75d8c9ed723993ae88212cc`.
- Teacher: `Alibaba-NLP/gte-modernbert-base` at `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, normalized `last_hidden_state[:,0]`.
- Fresh roles: TRAIN `[348000,360000)`, development `[360000,369000)`, buffer `[369000,370000)`, official `[370000,400000)`.
- Architectures: geodesic control; Euclidean residual `normalize(weighted_anchor + residual)`; free Euclidean `normalize(predicted_vector)`.
- Every arm had 123,616,258 parameters and used tensor-identical TRAIN cache rows and minibatch schedules.
- Nine fresh models: three architectures × seeds 424242, 271828, 314159; fixed final step 2400, no checkpoint selection.
- Protocol and hashes were frozen before development in `PREDEVELOPMENT_FREEZE.json`; the winner, evaluator, attacks, and official policy were frozen before official access in `OFFICIAL_PREACCESS_FREEZE.json`.

## TRAIN and development architecture comparison

All nine fixed runs completed. The direct-safe development results were:

| architecture | cosine mean | strongest AUC mean | rank mean | decoder child cosine | TRAIN-probe child cosine |
|---|---:|---:|---:|---:|---:|
| geodesic control | 0.946881 | 0.798123 | 82.92 | 0.884914 | 0.896716 |
| Euclidean residual | 0.987009 | 0.697732 | 119.96 | 0.912976 | 0.903123 |
| free Euclidean | 0.853888 | 0.999791 | 49.67 | 0.814200 | 0.808889 |

The geodesic aggregate is degraded by a predeclared, retained failure: seed 314159 saturated the 0.5-radian cap at every evaluated merge, yielding cosine 0.866647, rank 8.86, and AUC 1.0. Its other two seeds were healthy (cosine 0.986975/0.987019; AUC 0.699451/0.694918).

On the two healthy matched seed pairs, Euclidean residual changed cosine by only +0.0000438 on average and improved AUC by only 0.0002377, far below the frozen 0.02 meaningful-improvement threshold. Free Euclidean failed fidelity, rank, and AUC gates.

**Selection:** Euclidean residual won because numerical validity was the first frozen priority and it alone passed every gate for all three seeds. It did not win by detectability reduction. No tuning or protocol revision followed development.

## One-shot official result

The official source was marked spent before loading. Selected Euclidean-residual results:

| seed | cosine (95% bootstrap CI) | logistic AUC (95% group CI) | RBF AUC | rank | decoder child cosine | TRAIN-probe child cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 424242 | 0.987442 [0.987127, 0.987796] | 0.694686 [0.686022, 0.703791] | 0.599539 | 119.40 | 0.911177 | 0.901124 |
| 271828 | 0.987533 [0.987220, 0.987883] | 0.693427 [0.685408, 0.701197] | 0.603794 | 119.52 | 0.909887 | 0.899855 |
| 314159 | 0.987407 [0.987079, 0.987767] | 0.696824 [0.689109, 0.705158] | 0.612139 | 119.38 | 0.911620 | 0.901433 |
| **mean** | **0.987461** | **0.694979** | **0.605157** | **119.44** | **0.910895** | **0.900804** |

Per-corpus strongest AUCs were about 0.624–0.628 (FineWeb-Edu), 0.606–0.615 (FineWeb), 0.628–0.638 (OpenWebText), and 0.737–0.740 (Wikipedia). Wikipedia remains the easiest corpus to distinguish.

All official outputs were finite and unit-normalized; all TRAIN-probe floors passed. Full recursion covered 4,000 documents, 11,329 leaves, 4,812 internal merges, and depth up to 4. Euclidean arms have no angular cap; cap-saturation telemetry was exactly zero. Mean anchor-to-output angle was about 0.0347 radians, and final cosine improved over the weighted anchor by only about 0.00014.

V3's separate official source produced mean logistic AUC 0.69205. V4's 0.69498 is not a paired cross-version comparison, but it plainly does not support a practically important reduction. The exact matched V4 development comparison reaches the same conclusion.

## Downstream tasks

Public-benchmark teacher-only leaf tensors were reused hash-identically from V3 because teacher, revision, pooling, and chunking were unchanged; no V3 learned merger output was reused. This makes V3/V4 downstream comparisons more tightly matched. `DOWNSTREAM_AUDIT.json` verifies every cache file hash, standard metrics, and exclusions.

Three-seed selected-model means:

- NFCorpus: nDCG@10 0.348170, Recall@10 0.168303, Recall@100 0.317527, MRR@10 0.550243. Direct-teacher nDCG@10 was 0.348292.
- LongEmbed NarrativeQA: nDCG@10 0.549619, Recall@10 0.667145, Recall@100 0.890420, MRR@10 0.512587. Token-weighted mean remained better at nDCG@10 0.560881.
- LexGLUE ECtHR-A: macro F1 0.531362, micro F1 0.643946, samples F1 0.564356, subset accuracy 0.454000.
- 20 Newsgroups clustering: ARI 0.461931, NMI 0.607302. Unweighted mean remained better (ARI 0.472241, NMI 0.614374).

Direct full-document teacher classification/clustering was excluded under the frozen identical-sample policy because each dataset had context-overflow rows; no eligible subset was silently substituted. NFCorpus included the direct teacher because every required row fit. NarrativeQA omitted it because full documents exceed context.

## Interpretation

Three observations are robust:

1. A weighted child anchor already captures nearly all attainable cosine; the learned correction is small.
2. Euclidean residual removes the geodesic cap failure mode, so geometry affects numerical robustness.
3. Detectability remains around 0.695 despite removing spherical machinery, while unconstrained whole-vector prediction collapses scientifically.

Thus V3's signature is better explained by the harder problem of recursively predicting a whole-span CLS representation from child CLS representations than by geodesic constraints themselves. A future experiment should target missing cross-child information or a teacher objective explicitly designed for compositional sufficiency, rather than another coordinate-system change.

## Artifacts

Machine-readable source: `results/v4_euclidean/FINAL_METRICS_SUMMARY.json`. Official per-seed outputs, development comparisons, downstream outputs, audit sidecars, publication CSVs, and PNG/PDF figures are under `results/v4_euclidean/` and `figures/v4_euclidean/`.
