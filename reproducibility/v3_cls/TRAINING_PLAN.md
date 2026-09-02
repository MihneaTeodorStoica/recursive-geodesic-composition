# V3 CLS fresh-training plan (frozen before data access)

## Scientific status
V3 is a new experiment, not a reinterpretation of V1/V2/V2.2. It uses the V1 final K=4 strict-decoder architecture but replaces the foundation teacher representation with the documented normalized CLS embedding from pinned `Alibaba-NLP/gte-modernbert-base` revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`.

## Data allocation
Per corpus and pinned revision, half-open source-index ranges are frozen in `SOURCE_ALLOCATION.json`: TRAIN `[296000,308000)`, development `[308000,317000)`, unused buffer `[317000,318000)`, future one-shot official `[318000,348000)`. These begin strictly after the highest protected V2.2 range end `[266000,296000)`.

## Teacher extraction
`teacher_pooling: cls` is mandatory. Implementation: tokenize raw leaf/span token ids by re-adding tokenizer CLS/SEP special tokens, run the frozen AutoModel with `trust_remote_code=True`, select `outputs.last_hidden_state[:,0]`, cast to float, and L2-normalize. V1 `masked_mean` remains available only for exact historical reproduction and is rejected by V3 guards.

## Initialization
All V3 merger/decoder parameters are initialized from random seed only. Learned V1/V2/V2.2 checkpoints, autoencoders, probes, caches, and tuned states are forbidden by config and runtime guards.

## Final seeds
`424242`, `271828`, `314159`. The same frozen protocol is used for all final seeds; no seed replacement or cherry-picking.

## Inherited V1 architecture choices
K=4 contiguous recursive reduction, 1024-token leaves, hidden size 1024, 8 encoder layers, 16 heads, tangent-expmap geodesic parameterization, max angle 0.5, semantic importance, strict parent-only decoder, decoder length mode predicted, AdamW, cosine learning-rate schedule, no mixed precision, corpus-balanced batches, paired teacher cosine plus mean/variance distribution losses and strict decoder reconstruction/importance/cycle losses.

## Newly specified V3 choices
Because the historical V1 continuation initialization is missing, V3 trains the merger before decoder pressure becomes dominant: no encoder freeze; decoder reconstruction starts at step 300 and ramps over 300 steps. Final TRAIN-only models run 1200 steps, batch size 8, LR `2e-5` with warmup 50 and min LR `2e-6`, weight decay 0.001, distribution mean weight 0.25, variance weight 256, reconstruction weight 0.15, cycle weight 5.0, importance loss 0.025, importance prior/delta/sharpness/slot 0.01 each. Checkpoint rule is fixed final TRAIN-only checkpoint before development access.

## Pilot and amendment rules
A TRAIN-only pilot may use the same teacher/ranges with tiny model/count solely for engineering and numerical health. If TRAIN-only objectives are numerically pathological, an amendment must be written before development or official data access, preserving failed runs and applying one identical amended protocol to all three seeds.

## Health criteria
Require finite losses/vectors, nontrivial generated-vector effective rank, no invalid token-mass propagation, geodesic saturation diagnostics recorded, paired cosine improving on TRAIN, and decoder child reconstruction not NaN. Development is used only after TRAIN cache/protocol creation and is considered spent if materially used for model selection.
