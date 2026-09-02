# V3 protocol amendment V2 — bounded late linear-adversarial alignment candidate

Status: predeclared after development round 1 and before development round 2 access.

## Round-1 diagnosis (spent development population)
Across all three Amendment-V1 seeds, normalized CLS paired cosine was stable at 0.98709–0.98717 and recursion/probe health passed, but the strongest orientation-safe attack was consistently linear logistic AUC 0.6947–0.6977 while RBF AUC was only 0.5951–0.5977. Generated variance was already close to teacher variance (mean variance ratio 1.0018–1.0036), whereas a small but coherent mean-vector difference remained (L2 0.01155–0.01186). Covariance relative Frobenius error was ~0.103. This pattern supports testing a direct linear distribution discriminator rather than naive minibatch covariance matching.

## One bounded candidate
Train exactly one new fresh-from-random seed-424242 candidate for the same 2400 steps and architecture. All Amendment-V1 settings remain identical except a linear discriminator is activated only from step 1201, after merger pretraining, with adversarial alignment weight 0.05, discriminator AdamW LR `1e-4`, weight decay `1e-3`, one discriminator update per merger update, and minimum 8 paired samples. The merger receives the existing gradient-reversal/fooling objective. This is one hypothesis test, not a sweep; no alternative weights will be tried.

## Independent selection population
Round 1 accessed source rows only through index 313998 at maximum. Development round 2 is preallocated to the previously untouched source subrange `[314000,317000)` and selects the first 400 eligible documents per corpus. It is used once.

## Mechanical selection rule
Select V2 only if all hold on round 2: (1) paired cosine >= 0.985; (2) strongest orientation-safe AUC is at least 0.03 below the round-1 Amendment-V1 seed-424242 AUC 0.6951085; (3) pooled built-in decoder and TRAIN-fitted probe child cosine each decline by no more than 0.01 relative to round-1 seed 424242; (4) effective rank >= 100; (5) no nonfinite value or geodesic saturation. Otherwise retain Amendment V1. If selected, discard no evidence, retrain all three final seeds from random under identical V2 settings, refit TRAIN-only probes, and freeze a new official candidate set. Official data remains inaccessible throughout.
