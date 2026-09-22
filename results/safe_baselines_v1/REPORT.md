# Matched safe-arm baselines

Additive post-review analysis of the previously evaluated 4,000 documents. All methods use the same regenerated FP32 leaves and direct teacher targets; no merger retraining.

| Representation | Teacher cosine | Delta vs weighted | Delta vs unweighted | Source AUC [95% CI] | ms/document |
| --- | ---: | ---: | ---: | --- | ---: |
| First chunk | 0.958453 | -0.024485 | +0.001520 | 0.744883 [0.735426, 0.755457] | 0.000241 |
| Unweighted mean | 0.956933 | -0.026005 | +0.000000 | 0.899689 [0.892123, 0.907488] | 0.000875 |
| Token-weighted mean | 0.982938 | +0.000000 | +0.026005 | 0.833522 [0.824367, 0.842743] | 0.000720 |
| Learned reference | 0.983175 | +0.000237 | +0.026242 | 0.830841 [0.822036, 0.840090] | 0.078833 |
| Final learned parent | 0.985045 | +0.002107 | +0.028113 | 0.584294 [0.578340, 0.590430] | 0.078833 |

## Paired teacher-cosine contrasts

| Contrast | Mean difference | Paired 95% CI | Fraction positive |
| --- | ---: | --- | ---: |
| parent_minus_reference | +0.001870762 | [+0.001792600, +0.001950594] | 0.8075 |
| parent_minus_unweighted | +0.028112702 | [+0.026939146, +0.029312151] | 0.9657 |
| parent_minus_weighted | +0.002107322 | [+0.002021373, +0.002193885] | 0.8140 |
| reference_minus_weighted | +0.000236560 | [+0.000201308, +0.000273966] | 0.5707 |

## Shuffled-target calibration

Within-corpus derangements, no self-pairings; these are scale diagnostics, not competitive baselines or downstream retrieval results.

- First chunk: 0.511931
- Unweighted mean: 0.529562
- Token-weighted mean: 0.530287
- Learned reference: 0.529882
- Final learned parent: 0.527467

## Provenance and interpretation

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Timing: device_resident_leaves_to_device_resident_parent; float32; batch size 64; 3 warm-up and 20 measured passes; TF32 disabled.
- Reference and parent timing use the same measured full-forward call; reference-only speed is not claimed.
- No teacher encoding, cache loading, transfers, decoder evaluation, or classifier fitting is included in these inference timings.
- Original official results remain unchanged. Regenerated tensor byte identity is not claimed; preparation uses safe-only rather than full-document leaf batches.
- Confidence intervals condition on fixed checkpoints, fitted classifiers, split, and selected classifier families. They do not establish downstream utility or uncertainty over arbitrary training seeds.

## Comparison with published learned results

Tolerance checks diagnose numerical reproducibility, not scientific acceptance; all differences are retained.

| Seed | Scope | Cosine difference | AUC difference | Within tolerances |
| --- | --- | ---: | ---: | --- |
| 271828 | fineweb | +0.000001763 | -0.000576000 | False |
| 271828 | fineweb_edu | +0.000004914 | +0.000138000 | True |
| 271828 | openwebtext | -0.000003315 | -0.000014000 | True |
| 271828 | pooled | +0.000000780 | +0.000047000 | True |
| 271828 | wikipedia_en_20231101 | -0.000000243 | +0.000074000 | True |
| 314159 | fineweb | +0.000001406 | -0.000328000 | True |
| 314159 | fineweb_edu | +0.000004673 | +0.000292000 | True |
| 314159 | openwebtext | -0.000003155 | +0.000368000 | True |
| 314159 | pooled | +0.000000494 | +0.000004250 | True |
| 314159 | wikipedia_en_20231101 | -0.000000946 | -0.000016000 | True |
| 424242 | fineweb | +0.000001773 | -0.000148000 | True |
| 424242 | fineweb_edu | +0.000004857 | +0.000306000 | True |
| 424242 | openwebtext | -0.000003133 | +0.000326000 | True |
| 424242 | pooled | +0.000000746 | +0.000133875 | True |
| 424242 | wikipedia_en_20231101 | -0.000000511 | +0.000062000 | True |
