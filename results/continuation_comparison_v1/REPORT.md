# Matched source-versus-final continuation comparison

The released initialization-source merger versus its fixed step-400 clean-v6 final checkpoint, across all three lineage-matched pairs. This evaluates the continuation bundle as a whole, not a decoder-objective ablation or training from scratch.

| Metric (documents) | Source | Final | Final minus source [paired 95% CI] |
| --- | ---: | ---: | --- |
| Teacher cosine (4,000) | 0.985044633 | 0.985045479 | +0.000000847 [+0.000000610, +0.000001081] |
| Source AUC (2,000 held out) | 0.584827167 | 0.584293875 | -0.000533292 [-0.000675928, -0.000392327] |
| Parent-copy cosine (606) | 0.914068899 | 0.914021511 | -0.000047389 [-0.000048109, -0.000046613] |
| Probe cosine (606) | 0.908114434 | 0.908105018 | -0.000009416 [-0.000011671, -0.000007102] |
| Probe gain over copy (606) | -0.005954465 | -0.005916492 | +0.000037973 [+0.000035859, +0.000040216] |
| Probe order gap (606) | 0.000910307 | 0.000908435 | -0.000001872 [-0.000003767, +0.000000202] |

## Parent-vector diagnostics

- Source/final parent agreement: +0.999999912 [+0.999999910, +0.999999915].
- Source minus weighted pooling: +0.002106475 [+0.002019039, +0.002196267].
- Final minus weighted pooling: +0.002107322 [+0.002019842, +0.002197297].
- Fraction of documents with positive checkpoint-mean teacher change: 0.565750000.
- Fraction of exact-four documents with positive checkpoint-mean probe gain change: 0.922442244.

## Teacher fidelity: all checkpoints and prespecified strata

| Checkpoint | Scope | Documents | Source | Final | Change [95% CI] | Fraction positive |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 271828 | pooled | 4000 | 0.985032000 | 0.985033588 | +0.000001588 [+0.000001222, +0.000001951] | 0.598750 |
| 271828 | fineweb_edu | 1000 | 0.985144920 | 0.985146033 | +0.000001112 [+0.000000376, +0.000001865] | 0.590000 |
| 271828 | fineweb | 1000 | 0.983543761 | 0.983545370 | +0.000001608 [+0.000000606, +0.000002556] | 0.587000 |
| 271828 | openwebtext | 1000 | 0.986670613 | 0.986672560 | +0.000001948 [+0.000001264, +0.000002647] | 0.603000 |
| 271828 | wikipedia_en_20231101 | 1000 | 0.984768706 | 0.984770388 | +0.000001683 [+0.000000994, +0.000002386] | 0.615000 |
| 271828 | arity_2 | 2707 | 0.987834814 | 0.987836370 | +0.000001556 [+0.000001161, +0.000001930] | 0.635759 |
| 271828 | arity_3 | 687 | 0.980565042 | 0.980565676 | +0.000000634 [-0.000000198, +0.000001464] | 0.509461 |
| 271828 | arity_4 | 606 | 0.977575867 | 0.977578677 | +0.000002810 [+0.000001263, +0.000004513] | 0.534653 |
| 314159 | pooled | 4000 | 0.985052262 | 0.985052647 | +0.000000386 [-0.000000022, +0.000000811] | 0.498750 |
| 314159 | fineweb_edu | 1000 | 0.985182983 | 0.985183612 | +0.000000629 [-0.000000095, +0.000001407] | 0.495000 |
| 314159 | fineweb | 1000 | 0.983556711 | 0.983558870 | +0.000002159 [+0.000001264, +0.000003104] | 0.557000 |
| 314159 | openwebtext | 1000 | 0.986692897 | 0.986693233 | +0.000000336 [-0.000000393, +0.000001052] | 0.500000 |
| 314159 | wikipedia_en_20231101 | 1000 | 0.984776456 | 0.984774875 | -0.000001581 [-0.000002472, -0.000000708] | 0.443000 |
| 314159 | arity_2 | 2707 | 0.987867935 | 0.987868449 | +0.000000514 [+0.000000141, +0.000000886] | 0.495013 |
| 314159 | arity_3 | 687 | 0.980547520 | 0.980547569 | +0.000000049 [-0.000001238, +0.000001349] | 0.519651 |
| 314159 | arity_4 | 606 | 0.977581517 | 0.977581714 | +0.000000197 [-0.000001482, +0.000001847] | 0.491749 |
| 424242 | pooled | 4000 | 0.985049637 | 0.985050203 | +0.000000566 [+0.000000232, +0.000000909] | 0.499000 |
| 424242 | fineweb_edu | 1000 | 0.985184306 | 0.985185347 | +0.000001042 [+0.000000507, +0.000001599] | 0.518000 |
| 424242 | fineweb | 1000 | 0.983570597 | 0.983572339 | +0.000001742 [+0.000001044, +0.000002501] | 0.548000 |
| 424242 | openwebtext | 1000 | 0.986640447 | 0.986640752 | +0.000000305 [-0.000000283, +0.000000911] | 0.491000 |
| 424242 | wikipedia_en_20231101 | 1000 | 0.984803198 | 0.984802375 | -0.000000824 [-0.000001551, -0.000000106] | 0.439000 |
| 424242 | arity_2 | 2707 | 0.987862716 | 0.987863619 | +0.000000903 [+0.000000589, +0.000001218] | 0.522349 |
| 424242 | arity_3 | 687 | 0.980566264 | 0.980565439 | -0.000000824 [-0.000001587, -0.000000050] | 0.425036 |
| 424242 | arity_4 | 606 | 0.977566259 | 0.977566897 | +0.000000638 [-0.000000891, +0.000002185] | 0.478548 |
| three_seed_mean | pooled | 4000 | 0.985044633 | 0.985045479 | +0.000000847 [+0.000000610, +0.000001081] | 0.565750 |
| three_seed_mean | fineweb_edu | 1000 | 0.985170736 | 0.985171664 | +0.000000928 [+0.000000478, +0.000001373] | 0.575000 |
| three_seed_mean | fineweb | 1000 | 0.983557023 | 0.983558860 | +0.000001836 [+0.000001346, +0.000002366] | 0.607000 |
| three_seed_mean | openwebtext | 1000 | 0.986667985 | 0.986668848 | +0.000000863 [+0.000000463, +0.000001271] | 0.576000 |
| three_seed_mean | wikipedia_en_20231101 | 1000 | 0.984782787 | 0.984782546 | -0.000000241 [-0.000000723, +0.000000214] | 0.505000 |
| three_seed_mean | arity_2 | 2707 | 0.987855155 | 0.987856146 | +0.000000991 [+0.000000795, +0.000001183] | 0.594385 |
| three_seed_mean | arity_3 | 687 | 0.980559608 | 0.980559561 | -0.000000047 [-0.000000662, +0.000000568] | 0.491994 |
| three_seed_mean | arity_4 | 606 | 0.977574548 | 0.977575763 | +0.000001215 [+0.000000163, +0.000002321] | 0.521452 |

## Source detectability: fixed logistic/RBF classifier recipe

Two fresh source classifiers per checkpoint/scope (30 fits); all final classifiers reused with file-hash checks from improvement 1. Identical document-grouped 50:50 splits. Strongest observed family is fixed within resampling, independently for each stage/checkpoint.

| Checkpoint | Scope | Held-out documents | Source AUC | Final AUC | Change [95% CI] | Source / final selected family |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 424242 | fineweb | 500 | 0.574094000 | 0.573656000 | -0.000438000 [-0.000852050, -0.000023950] | standard_scaled_logistic / standard_scaled_logistic |
| 271828 | fineweb | 500 | 0.570440000 | 0.569590000 | -0.000850000 [-0.001384050, -0.000331950] | standard_scaled_logistic / standard_scaled_logistic |
| 314159 | fineweb | 500 | 0.573352000 | 0.572516000 | -0.000836000 [-0.001332000, -0.000324000] | standard_scaled_logistic / standard_scaled_logistic |
| three_seed_mean | fineweb | 500 | 0.572628667 | 0.571920667 | -0.000708000 [-0.000998067, -0.000399317] | fixed per checkpoint |
| 424242 | fineweb_edu | 500 | 0.562496000 | 0.562254000 | -0.000242000 [-0.000716050, +0.000232200] | standard_scaled_logistic / standard_scaled_logistic |
| 271828 | fineweb_edu | 500 | 0.567900000 | 0.567450000 | -0.000450000 [-0.000990100, +0.000098100] | standard_scaled_logistic / standard_scaled_logistic |
| 314159 | fineweb_edu | 500 | 0.569402000 | 0.569982000 | +0.000580000 [-0.000016050, +0.001176100] | standard_scaled_logistic / standard_scaled_logistic |
| three_seed_mean | fineweb_edu | 500 | 0.566599333 | 0.566562000 | -0.000037333 [-0.000371350, +0.000272000] | fixed per checkpoint |
| 424242 | openwebtext | 500 | 0.587120000 | 0.586988000 | -0.000132000 [-0.000576050, +0.000284050] | standard_scaled_logistic / standard_scaled_logistic |
| 271828 | openwebtext | 500 | 0.581576000 | 0.581400000 | -0.000176000 [-0.000714150, +0.000364100] | standard_scaled_logistic / standard_scaled_logistic |
| 314159 | openwebtext | 500 | 0.583238000 | 0.582800000 | -0.000438000 [-0.001004050, +0.000126050] | standard_scaled_logistic / standard_scaled_logistic |
| three_seed_mean | openwebtext | 500 | 0.583978000 | 0.583729333 | -0.000248667 [-0.000576083, +0.000072683] | fixed per checkpoint |
| 424242 | pooled | 2000 | 0.583379250 | 0.582942625 | -0.000436625 [-0.000619500, -0.000237225] | standard_scaled_logistic / standard_scaled_logistic |
| 271828 | pooled | 2000 | 0.582905000 | 0.582171500 | -0.000733500 [-0.000969256, -0.000491494] | standard_scaled_logistic / standard_scaled_logistic |
| 314159 | pooled | 2000 | 0.588197250 | 0.587767500 | -0.000429750 [-0.000667769, -0.000211212] | standard_scaled_logistic / standard_scaled_logistic |
| three_seed_mean | pooled | 2000 | 0.584827167 | 0.584293875 | -0.000533292 [-0.000675928, -0.000392327] | fixed per checkpoint |
| 424242 | wikipedia_en_20231101 | 500 | 0.641342000 | 0.641918000 | +0.000576000 [+0.000085800, +0.001062050] | standard_scaled_logistic / standard_scaled_logistic |
| 271828 | wikipedia_en_20231101 | 500 | 0.632490000 | 0.632154000 | -0.000336000 [-0.000800050, +0.000112050] | standard_scaled_logistic / standard_scaled_logistic |
| 314159 | wikipedia_en_20231101 | 500 | 0.636056000 | 0.636116000 | +0.000060000 [-0.000482050, +0.000622250] | standard_scaled_logistic / standard_scaled_logistic |
| three_seed_mean | wikipedia_en_20231101 | 500 | 0.636629333 | 0.636729333 | +0.000100000 [-0.000208683, +0.000409450] | fixed per checkpoint |

## Reconstruction: all checkpoints and corpora

| Checkpoint | Scope | Source probe | Final probe | Probe change [95% CI] | Source gain over copy | Final gain over copy | Gain change [95% CI] | Order-gap change [95% CI] |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 271828 | pooled | 0.908068030 | 0.908058086 | -0.000009944 [-0.000013581, -0.000006228] | -0.006021913 | -0.005971144 | +0.000050768 [+0.000047065, +0.000054540] | -0.000003309 [-0.000006627, +0.000000072] |
| 271828 | fineweb_edu | 0.903027048 | 0.903004327 | -0.000022721 [-0.000030834, -0.000014592] | -0.006531665 | -0.006489772 | +0.000041894 [+0.000033874, +0.000049680] | -0.000010922 [-0.000018330, -0.000003526] |
| 271828 | fineweb | 0.902920847 | 0.902914296 | -0.000006551 [-0.000013832, +0.000000244] | -0.005879859 | -0.005824738 | +0.000055121 [+0.000047524, +0.000062103] | +0.000000131 [-0.000006058, +0.000005870] |
| 271828 | openwebtext | 0.909323501 | 0.909321772 | -0.000001729 [-0.000009141, +0.000005511] | -0.005672013 | -0.005612037 | +0.000059976 [+0.000052495, +0.000067458] | +0.000001798 [-0.000005051, +0.000008308] |
| 271828 | wikipedia_en_20231101 | 0.914911548 | 0.914903096 | -0.000008452 [-0.000015265, -0.000002080] | -0.005984402 | -0.005936765 | +0.000047637 [+0.000040799, +0.000054007] | -0.000003616 [-0.000010016, +0.000002609] |
| 314159 | pooled | 0.908160261 | 0.908155469 | -0.000004792 [-0.000007904, -0.000001614] | -0.005905624 | -0.005872802 | +0.000032822 [+0.000029864, +0.000035993] | -0.000000170 [-0.000002828, +0.000002717] |
| 314159 | fineweb_edu | 0.903045343 | 0.903036479 | -0.000008863 [-0.000015601, -0.000001885] | -0.006442470 | -0.006410374 | +0.000032096 [+0.000025781, +0.000038696] | -0.000001811 [-0.000007437, +0.000004242] |
| 314159 | fineweb | 0.902982433 | 0.902977003 | -0.000005431 [-0.000011370, +0.000000599] | -0.005893525 | -0.005862279 | +0.000031246 [+0.000025413, +0.000037041] | -0.000003746 [-0.000008937, +0.000001534] |
| 314159 | openwebtext | 0.909553926 | 0.909546402 | -0.000007524 [-0.000013461, -0.000001196] | -0.005645380 | -0.005615314 | +0.000030066 [+0.000023967, +0.000036330] | -0.000003687 [-0.000008893, +0.000001605] |
| 314159 | wikipedia_en_20231101 | 0.914978615 | 0.914979699 | +0.000001084 [-0.000004068, +0.000006806] | -0.005682184 | -0.005645490 | +0.000036693 [+0.000031323, +0.000042355] | +0.000006487 [+0.000001688, +0.000011764] |
| 424242 | pooled | 0.908115011 | 0.908101500 | -0.000013511 [-0.000015838, -0.000011118] | -0.005935860 | -0.005905532 | +0.000030328 [+0.000028046, +0.000032688] | -0.000002137 [-0.000004221, -0.000000057] |
| 424242 | fineweb_edu | 0.903233806 | 0.903217555 | -0.000016251 [-0.000021193, -0.000010764] | -0.006421937 | -0.006394777 | +0.000027160 [+0.000022362, +0.000032235] | -0.000001918 [-0.000006249, +0.000002916] |
| 424242 | fineweb | 0.902926741 | 0.902912486 | -0.000014255 [-0.000019647, -0.000009201] | -0.005947110 | -0.005917009 | +0.000030101 [+0.000024798, +0.000035214] | -0.000003487 [-0.000007923, +0.000000893] |
| 424242 | openwebtext | 0.909293770 | 0.909279824 | -0.000013946 [-0.000018438, -0.000009284] | -0.005662612 | -0.005631343 | +0.000031268 [+0.000026897, +0.000035920] | -0.000003622 [-0.000007595, +0.000000414] |
| 424242 | wikipedia_en_20231101 | 0.914918695 | 0.914908280 | -0.000010416 [-0.000014695, -0.000005981] | -0.005746600 | -0.005714274 | +0.000032326 [+0.000028092, +0.000036884] | -0.000000182 [-0.000004077, +0.000003725] |
| three_seed_mean | pooled | 0.908114434 | 0.908105018 | -0.000009416 [-0.000011671, -0.000007102] | -0.005954465 | -0.005916492 | +0.000037973 [+0.000035859, +0.000040216] | -0.000001872 [-0.000003767, +0.000000202] |
| three_seed_mean | fineweb_edu | 0.903102065 | 0.903086120 | -0.000015945 [-0.000021022, -0.000010560] | -0.006465358 | -0.006431641 | +0.000033717 [+0.000028897, +0.000038856] | -0.000004884 [-0.000009236, -0.000000107] |
| three_seed_mean | fineweb | 0.902943340 | 0.902934595 | -0.000008745 [-0.000013273, -0.000004723] | -0.005906831 | -0.005868008 | +0.000038823 [+0.000034575, +0.000042459] | -0.000002367 [-0.000005779, +0.000000875] |
| three_seed_mean | openwebtext | 0.909390399 | 0.909382666 | -0.000007733 [-0.000012031, -0.000003192] | -0.005660002 | -0.005619565 | +0.000040437 [+0.000035969, +0.000044903] | -0.000001837 [-0.000005609, +0.000002056] |
| three_seed_mean | wikipedia_en_20231101 | 0.914936286 | 0.914930358 | -0.000005928 [-0.000009724, -0.000001947] | -0.005804395 | -0.005765510 | +0.000038885 [+0.000034991, +0.000042870] | +0.000000896 [-0.000002727, +0.000004541] |

## Scope and uncertainty

- Paired endpoint comparison: pretrained source merger, discarded source decoder, freshly initialized decoder, 300 decoder-only updates, then 100 joint updates with encoder LR scale 0.02. It cannot isolate the decoder loss, distribution penalties, schedule, or other training components.
- Released source/candidate probes share documented architecture, initialization and TRAIN fitting recipe. Their TRAIN selection and minibatch schedule metadata are checked; no probes are fitted here. This is not a comparison with the distinct historical matched-pooling probes.
- 2,000 shared document bootstrap resamples, paired across stages and checkpoints. Scalar checkpoint scores are averaged within documents; AUCs are computed per checkpoint and averaged, never from averaged classifier scores. Fixed classifier family choices; no classifier refitting or family reselection in bootstrap.
- AUC bootstrap counts weight both observations of each teacher/generated document pair. Orientation-safe AUC is recomputed in each resample under the original convention. This is conditional uncertainty, not a simultaneous multiple-comparison guarantee or new-training-seed uncertainty.
- All methods use the same regenerated inputs. Final parent/probe tensors and final classifier scores are reused unchanged. No original embedding-byte identity is asserted. These already evaluated documents are not a fresh confirmatory test.
- Child cosine, gain over copy, and the order gap are local diagnostics, not downstream utility, losslessness, or document-specific order storage. No threshold crossing or favorable checkpoint is used for model selection.
- No merger, decoder, or probe updates; no teacher encoding. Both source classifier families, all checkpoints, corpora, and arities are retained regardless of sign.
