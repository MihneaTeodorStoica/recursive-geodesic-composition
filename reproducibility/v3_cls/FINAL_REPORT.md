# V3 CLS final report

Status: **complete; independent audit PASS; ready for human review and a future proposed `v3-paper` release (not published).**

## Lineage and teacher
Branch `v3-paper-cls` descends from V1 publication commit `c6b3f28830e7965533e1d036c2a749836b7848b6`. V3 uses pinned `Alibaba-NLP/gte-modernbert-base@e7f32e3c00f91d699e8c43b53106206bcc72bb22`, `last_hidden_state[:,0]`, then L2 normalization. Direct reference validation was tensor-identical (`max_abs_diff=0`). No learned V1/V2/V2.2 state or cache was loaded.

## Final protocol
Fresh random K=4 V1-final geodesic/strict-decoder architecture, seeds 424242/271828/314159, 2,400 AdamW steps, batch 8, LR 1e-4, warmup 160, cosine floor 1e-5. Steps 1–1200 train merger with paired cosine and corpus-balanced mean/variance distribution terms; decoder/reconstruction/cycle/importance losses ramp over steps 1201–1600 and remain joint through 2400. Amendment V2 added a linear adversarial discriminator from step 1201, weight 0.05, LR 1e-4. Selection was fixed final TRAIN-only step.

TRAIN `[296000,308000)`, development round 1 `[308000,...actual max 313998]`, fresh development round 2 `[314000,317000)`, buffer `[317000,318000)`, official `[318000,348000)`. The final three runs took 376/377/375 seconds (1,128 s total, 18.8 min); all histories were finite with zero geodesic saturation.

Checkpoint SHA-256:
- 424242: `5f44f72a48d64830dc411e8f5559c9a89a9737fcabd8a6bf6bd37bff9fc5c0d8`
- 271828: `3f9efe8cfdabd0d8997319969581fa2220c123c27abff3848ce1e0e7a0769fec`
- 314159: `9677f5ac5078856457d91ca0df2ddc222c76a2752bcb698cbd570e7ed1bebbc7`

## Development
Fresh round-2 final three-seed means: cosine **0.986939**, logistic AUC **0.651969**, RBF AUC **0.584439**, effective rank **121.64**, decoder child cosine **0.912438**, TRAIN-probe child cosine **0.902545**. Every probe floor passed and recursion was finite. Full seed/per-corpus values are in `results/v3_cls/FINAL_METRICS_SUMMARY.json`.

The bounded late-adversarial candidate formally passed its predeclared cross-round rule, but on the same round-2 population its AUC was 0.651691 versus 0.650813 for the non-adversarial control. Thus adversarial alignment showed no supported causal benefit; it was retained only to respect the frozen rule. No further search occurred.

## Official one-shot results
Official access was marked spent only after commit `1bf1cd7` froze checkpoints, probes, code, attacks, bootstrap, schemas, and semantics. No recovery was needed.

| Seed | cosine (95% bootstrap CI) | logistic AUC (95% CI) | RBF AUC |
|---|---|---|---|
| 424242 | 0.987013 [0.986728, 0.987298] | 0.691513 [0.684104, 0.701095] | 0.602185 |
| 271828 | 0.986954 [0.986668, 0.987238] | 0.690579 [0.683476, 0.699669] | 0.601159 |
| 314159 | 0.987041 [0.986751, 0.987339] | 0.694055 [0.686995, 0.703188] | 0.598578 |

Three-seed means: cosine **0.987003**, logistic AUC **0.692049**, RBF AUC **0.600641**. Mean per-corpus strongest AUC was approximately FineWeb-Edu 0.638, FineWeb 0.613, OpenWebText 0.638, Wikipedia 0.779; mean cosine was approximately 0.98699/0.98741/0.98853/0.98509 respectively. Wikipedia is the dominant negative result.

Full arm represented all tokens without root-teacher fabrication: 4,000 documents, maximum **79,674 tokens**, **78 leaves**, depth **4**, 4,891 internal merges; all finite, no geodesic or importance saturation. Official decoder/probe child cosine means were **0.913551/0.903537**, with all TRAIN floors passing. Safe effective rank mean was 117.33.

## Downstream
### Retrieval
NFCorpus (3,633 docs/323 test queries) recursive mean: nDCG@10 **0.348170**, Recall@10 **0.168303**, Recall@100 **0.317527**, MRR@10 **0.550243**. Weighted mean nDCG@10 0.348162; direct teacher 0.348292.

Long benchmark: **LongEmbed NarrativeQA**, revision `10039a580487dacecf79db69166e17ace3ede392`, 355 documents, 10,449 queries, 30.4M tokens, 29,882 leaves, maximum 488,429 tokens/477 leaves/depth 5. Recursive seeds nDCG@10: **0.549882/0.551776/0.548926**; weighted and unweighted means were **0.560881/0.560153**. Recursive Recall@100 was 0.8906–0.8922. Direct teacher was correctly omitted. MLDR English was metadata-inspected but infeasible at 200k documents/984.9MB compressed under the exact-cache disk budget.

### ECtHR-A and clustering
ECtHR-A correct multilabel OvR micro/macro F1:
- recursive 424242: **0.6347/0.5081**
- recursive 271828: **0.6335/0.5083**
- recursive 314159: **0.6301/0.5342**
- weighted: 0.6359/0.5052; unweighted: 0.6349/0.5155.

20 Newsgroups NMI/ARI:
- recursive: **0.6133/0.4644**, **0.6133/0.4640**, **0.6134/0.4645**
- weighted: 0.5974/0.4636; unweighted: 0.6144/0.4722.

No task rows were dropped. Direct-teacher task metrics were omitted when any canonical row exceeded context. Downstream successful runtime was ~60.3 min total (retrieval ~25 min; classification/clustering 35.3 min).

## Interpretation and negative findings
Correct documented CLS extraction yields excellent paired fidelity and stable recursion, but substantially worse detectability than V1's masked-mean experiment. Small coherent 768-D residual structure remains exploitable by a linear probe; high cosine does not imply distribution identity. Late adversarial alignment did not improve same-population AUC. On truly long NarrativeQA documents, simple leaf means outperformed recursive composition in nDCG/MRR, although recursion slightly improved Recall@100.

## Audit and release readiness
`FINAL_AUDIT.json` passes branch/base lineage, CLS identity, fresh initialization, range separation, identical seed protocol, checkpoint/probe hashes, pre-official freeze, one-shot official access, no full-root teacher fabrication, tensor-identical downstream leaves, standard metrics, and no silent exclusions. Historical result namespaces were unchanged. `PROPOSED_V3_PAPER_RELEASE_MANIFEST.json` is prepared but not published. V3 is ready for human review.
