# V4 Euclidean architecture ablation plan (frozen before source access)

## Question
Does V3's spherical/geodesic output constraint cause the persistent generated-vs-direct-CLS signature, or is distinguishability inherent to recursively predicting a whole-span CLS vector from child CLS vectors?

## Teacher/data
Pinned `Alibaba-NLP/gte-modernbert-base@e7f32e3c00f91d699e8c43b53106206bcc72bb22`; documented `last_hidden_state[:,0]` then L2 normalization. Independent preaccess validation: exact equality. Per-corpus roles: TRAIN `[348000,360000)`, development `[360000,369000)`, unused buffer `[369000,370000)`, future official `[370000,400000)`. All arms consume the same create-only cache tensors, IDs, counts, teacher targets and Python-seeded minibatch row schedule.

## Architectures
All use ordered K=4 slots, child/count/log-count encoding, 1024 hidden width, 8-layer 16-head contextual Transformer, shared trunk/head shape, semantic mass correction, strict parent-only one-layer decoder, 768-D outputs and final L2 normalization. Each has exactly **123,616,258 parameters** (**111,538,433 merger-side**); capacity is exactly matched.

1. **Geodesic control** (`composition_geometry=geodesic`): V3 tangent projection and radial-tanh exponential-map correction inside max angle 0.5 radians.
2. **Euclidean residual**: raw token-mass/semantic-weighted Euclidean anchor `a=sum(w_i e_i)` (not normalized internally), unconstrained 768-D residual `Delta`, output `normalize(a+Delta)`. No projection, exp map, interpolation or cap.
3. **Free Euclidean**: ordered contextual child/count features enter the same trunk with a zero anchor channel; the head predicts arbitrary `z`, output `normalize(z)`. No anchor requirement, projection, exp map or cap. Its head is Xavier initialized to avoid a zero vector.

The same semantic mass and parent-only decoder regularizers remain meaningful and are retained. Geodesic cap/saturation has no optimization role for Euclidean arms and is omitted (telemetry is always non-saturated). Anchor-final angle and raw residual norm are diagnostic only.

## Initialization and training
Fresh random initialization for every arm/seed; learned V1/V2/V3 state is forbidden. Final seeds 424242, 271828, 314159. Matched V3-final budget: 2,400 AdamW steps, batch 8 with exactly two rows/corpus, LR 1e-4, warmup 160, cosine floor 1e-5, weight decay .001. Merger/cosine/mean/variance objectives from steps 1–1200; strict decoder reconstruction/cycle/importance ramps over 1201–1600; linear adversarial alignment weight .05 and discriminator LR 1e-4 starts at 1201. Fixed final step, no validation checkpoint selection.

Loss coefficients: teacher cosine 1; distribution mean .25; diagonal variance 256; reconstruction .15; cycle 5; importance .025; salience prior/delta/sharpness/slot .01. These have architecture-independent meanings.

## TRAIN-only pilot
Exactly one 80-step full-width merger-only run per architecture, seed 424242, identical cache/minibatch schedule, LR 1e-4/warmup10/floor1e-5. Pass if outputs finite and normalized, max grad norm <100, no rank collapse, and last-20 teacher loss below first-20. No sweep. If all pass, use the frozen full protocol unchanged.

## Development comparison/selection
Train all three seeds for all three architectures before development. Evaluate all nine on identical source-order 1025–4096-token safe rows and full documents. Report paired cosine, orientation-safe logistic/RBF AUC and CIs, per corpus/arity/depth, mean/diagonal variance/full covariance mismatch, effective rank/uniqueness, decoder/probe retention, recursion finite/throughput. Control additionally reports anchor-target and final-target cosine, required angle/cap fraction. Residual reports anchor/final target cosine, residual norms/directions. Free reports output-anchor angle and norm health.

Predeclared winner rule: reject any arm with nonfinite/collapse/saturation, cosine <0.982, effective rank <100, or probe-floor failure. Among valid arms, prioritize mean strongest AUC; require >=0.02 absolute improvement over geodesic to claim meaningful detectability benefit, while cosine may be at most .003 below control and decoder/probe cosine at most .01 below control. If several qualify, choose lowest AUC, then highest cosine, then retention. If none meaningfully improve, geodesic control remains selected. Development permits at most one documented stability-only revision; no architecture/weight sweep.

## Official/statistics
After architecture selection, the already-trained three selected fixed-final seeds are frozen (no retraining against development), TRAIN-only probes fitted, and evaluator/attacks/bootstrap/schema hashes frozen. Official `[370000,400000)` is one-shot. Report 400-replicate group bootstrap CIs, architecture paired per-document cosine differences on development, and bootstrap AUC CIs; no fabricated full-root teacher metric.

## Health/downstream
Safe/full two-arm semantics match V3. Full recursion has no truncation/caps and exact token propagation/singleton carry. Downstream uses NFCorpus, pinned LongEmbed NarrativeQA, ECtHR-A, and 20 Newsgroups with standard metrics and shared V4 leaf tensors; direct teacher only for complete benchmark coverage.
