# Direct composition versus recursive reuse

Additive post-review tree/active-arity stress test on the same 606 exact-four-child safe prefixes. All three frozen final checkpoints; no merger retraining, teacher encoding, or classifier fitting.

Direct: `f4(x1,x2,x3,x4)` (one merge, depth one).
Recursive: `f2(f2(x1,x2),f2(x3,x4))` (three merges, depth two). Each `f2` is the same K4 operator with two active leading slots. Counts are summed exactly at each node.

| Representation | Mean teacher cosine | Difference vs direct | ms/document |
| --- | ---: | ---: | ---: |
| Flat unweighted pooling | 0.968715 | -0.008861 | 0.000722 |
| Flat token-weighted pooling | 0.971502 | -0.006074 | 0.000597 |
| Direct four-child merge | 0.977576 | +0.000000 | 0.078448 |
| Recursive binary tree | 0.977114 | -0.000461 | 0.205201 |

## Paired contrasts

| Contrast | Mean | Paired 95% CI |
| --- | ---: | --- |
| direct_minus_unweighted | +0.008861223 | [+0.008284266, +0.009474612] |
| direct_minus_weighted | +0.006073605 | [+0.005814834, +0.006354552] |
| recursive_minus_direct | -0.000461309 | [-0.000536903, -0.000389936] |
| recursive_minus_unweighted | +0.008399914 | [+0.007811105, +0.009027730] |
| recursive_minus_weighted | +0.005612296 | [+0.005340519, +0.005899458] |

Direct–recursive output cosine: **0.999441401**, conditional 95% CI [0.999425055, 0.999456762].

## Per-document degradation

Loss is direct teacher cosine minus recursive teacher cosine; positive means recursion is worse. Statistics below average checkpoint scores within each document, not embeddings or 1,818 independent samples.

- Fraction degraded: 0.749175
- Fraction improved: 0.250825
- Loss quantiles (min, p05, median, p95, max): -0.004296684, -0.000990082, +0.000465059, +0.001938611, +0.004846910
- Fraction losing more than 0.001: 0.244224
- Fraction losing more than 0.005: 0.000000
- Fraction losing more than 0.01: 0.000000

## Per-checkpoint and per-corpus results

| Checkpoint aggregate | Scope | Documents | Direct | Recursive | Recursive minus direct [95% CI] | Output cosine |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 424242 | pooled | 606 | 0.977567 | 0.977143 | -0.000423874 [-0.000513120, -0.000337607] | 0.999402 |
| 424242 | fineweb_edu | 148 | 0.978494 | 0.978214 | -0.000279141 [-0.000450187, -0.000104782] | 0.999428 |
| 424242 | fineweb | 133 | 0.974648 | 0.974344 | -0.000303885 [-0.000515953, -0.000095881] | 0.999364 |
| 424242 | openwebtext | 142 | 0.978155 | 0.977564 | -0.000590475 [-0.000756772, -0.000431095] | 0.999427 |
| 424242 | wikipedia_en_20231101 | 183 | 0.978483 | 0.977984 | -0.000498856 [-0.000669871, -0.000333436] | 0.999390 |
| 271828 | pooled | 606 | 0.977579 | 0.977101 | -0.000478098 [-0.000555258, -0.000403659] | 0.999477 |
| 271828 | fineweb_edu | 148 | 0.978487 | 0.977981 | -0.000506111 [-0.000691093, -0.000336851] | 0.999460 |
| 271828 | fineweb | 133 | 0.974644 | 0.974307 | -0.000336713 [-0.000520014, -0.000153388] | 0.999455 |
| 271828 | openwebtext | 142 | 0.978188 | 0.977739 | -0.000448688 [-0.000586864, -0.000304852] | 0.999479 |
| 271828 | wikipedia_en_20231101 | 183 | 0.978504 | 0.977923 | -0.000581019 [-0.000709506, -0.000454172] | 0.999505 |
| 314159 | pooled | 606 | 0.977582 | 0.977100 | -0.000481956 [-0.000563098, -0.000404456] | 0.999445 |
| 314159 | fineweb_edu | 148 | 0.978482 | 0.978108 | -0.000374564 [-0.000509366, -0.000245953] | 0.999436 |
| 314159 | fineweb | 133 | 0.974722 | 0.974270 | -0.000451700 [-0.000666694, -0.000251041] | 0.999426 |
| 314159 | openwebtext | 142 | 0.978218 | 0.977637 | -0.000580786 [-0.000730333, -0.000432560] | 0.999469 |
| 314159 | wikipedia_en_20231101 | 183 | 0.978438 | 0.977924 | -0.000514109 [-0.000665603, -0.000360794] | 0.999447 |
| three_seed_mean | pooled | 606 | 0.977576 | 0.977114 | -0.000461309 [-0.000536903, -0.000389936] | 0.999441 |
| three_seed_mean | fineweb_edu | 148 | 0.978488 | 0.978101 | -0.000386606 [-0.000531525, -0.000247293] | 0.999441 |
| three_seed_mean | fineweb | 133 | 0.974671 | 0.974307 | -0.000364100 [-0.000550338, -0.000183716] | 0.999415 |
| three_seed_mean | openwebtext | 142 | 0.978187 | 0.977647 | -0.000539983 [-0.000682077, -0.000404487] | 0.999458 |
| three_seed_mean | wikipedia_en_20231101 | 183 | 0.978475 | 0.977944 | -0.000531328 [-0.000663535, -0.000400383] | 0.999447 |

## Original full-document population

This is descriptive accounting, not a new full-document fidelity test. All global token/leaf/depth histograms match the original frozen results.

| Full token range (inclusive) | Documents | Percentage | K4 depth counts |
| --- | ---: | ---: | --- |
| 1,025–2,048 | 2707 | 67.675 | {'1': 2707} |
| 2,049–4,096 | 924 | 23.100 | {'1': 924} |
| 4,097–8,192 | 288 | 7.200 | {'2': 288} |
| 8,193–16,384 | 61 | 1.525 | {'2': 61} |
| 16,385–32,768 | 15 | 0.375 | {'3': 15} |
| >32,768 | 5 | 0.125 | {'3': 5} |

- K4 depth counts: {'1': 3631, '2': 349, '3': 20}
- Documents with actual recursive reuse (depth >=2): 369/4000
- Strictly above documented 8,192-token context: 81/4000 (2.025%).
- Total: 9,142,043 tokens, 11,356 leaves, 4,789 merges per checkpoint.

| Corpus | Depth 1 | Depth 2 | Depth 3 | Above 8,192 tokens |
| --- | ---: | ---: | ---: | ---: |
| fineweb_edu | 909 | 85 | 6 | 27 |
| fineweb | 910 | 84 | 6 | 19 |
| openwebtext | 919 | 78 | 3 | 16 |
| wikipedia_en_20231101 | 893 | 102 | 5 | 19 |

## Scope and reproducibility

- This is a tree/recursive-reuse stress test, **not a causal estimate of depth alone**: active arity, grouping, intermediate representations and merge count change together.
- Every semantic comparison concerns the same 3,073–4,096-token safe prefix and teacher target. It does not measure fidelity of unrestricted document roots or arbitrary trees/depths.
- Pooling is flat over original leaves. Normalizing intermediate pooled vectors while keeping only token counts would define a different baseline.
- Checkpoints, caches, improvement-1 results and original official results remain unchanged. Direct and flat-pooling vectors are checked against the exact-four subset of improvement 1.
- Intervals use 2,000 shared-document resamples, conditional on the fixed checkpoints. Checkpoint-mean summaries average per-document scores before resampling.
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU; FP32; TF32 disabled; document batch size 64; both first-level pairs share a 2B-sized call, followed by a B-sized root call.
- Timing excludes cache I/O, transfers and teacher encoding; includes tree packing and all merge calls. Three warm-up and 20 measured passes; checkpoint medians averaged.
