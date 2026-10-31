# Parent-copy and target-slot permutation calibration

606 identity-matched exact-four safe prefixes; all three frozen final checkpoints, their built-in decoders, and the released candidate TRAIN-fitted probes. No new training or teacher encoding. Parents and children are reused from the preceding matched analyses.

| Prediction rule | Ordered child cosine | Difference vs learned-parent copy | All-24 shuffled cosine | Ordered minus shuffled |
| --- | ---: | ---: | ---: | ---: |
| Learned parent copied | 0.914021511 | +0.000000000 | 0.914021511 | +0.000000000 |
| Built-in strict decoder | 0.915064015 | +0.001042504 | 0.915063330 | +0.000000686 |
| Released TRAIN-fitted probe | 0.908105018 | -0.005916492 | 0.907196583 | +0.000908435 |
| Weighted pooling copied | 0.920365100 | +0.006343590 | 0.920365100 | +0.000000000 |
| Unweighted pooling copied | 0.922575508 | +0.008553997 | 0.922575508 | +0.000000000 |

## Paired reconstruction contrasts

| Contrast | Mean | Conditional paired 95% CI | Fraction positive |
| --- | ---: | --- | ---: |
| probe_minus_parent_copy | -0.005916492 | [-0.006150681, -0.005693017] | 0.018152 |
| probe_minus_unweighted_copy | -0.014470489 | [-0.014971656, -0.013980381] | 0.004950 |
| probe_minus_weighted_copy | -0.012260082 | [-0.012512763, -0.011993684] | 0.004950 |
| strict_minus_parent_copy | +0.001042504 | [+0.001025339, +0.001060005] | 1.000000 |
| strict_minus_unweighted_copy | -0.007511493 | [-0.007972192, -0.007050390] | 0.000000 |
| strict_minus_weighted_copy | -0.005301085 | [-0.005419445, -0.005185391] | 0.000000 |

## Exact order controls

All 24 target-slot permutations include the identity; each target lands in any slot with probability 1/4. The nine derangements exclude every original slot match. They are scored after inference, never fed to a decoder and never treated as independent documents.

| Method | Ordered minus all-24 [95% CI] | Ordered minus deranged-9 [95% CI] | Oracle best assignment | Oracle best minus ordered |
| --- | --- | --- | ---: | ---: |
| Released TRAIN-fitted probe | +0.000908435 [+0.000699193, +0.001115978] | +0.001211247 [+0.000932258, +0.001487970] | 0.911371109 | 0.003266091 |
| Learned parent copied | +0.000000000 [+0.000000000, +0.000000000] | +0.000000000 [+0.000000000, +0.000000000] | 0.914021511 | 0.000000000 |
| Built-in strict decoder | +0.000000686 [+0.000000596, +0.000000772] | +0.000000914 [+0.000000795, +0.000001030] | 0.915064957 | 0.000000942 |
| Unweighted pooling copied | +0.000000000 [+0.000000000, +0.000000000] | +0.000000000 [+0.000000000, +0.000000000] | 0.922575508 | 0.000000000 |
| Weighted pooling copied | +0.000000000 [+0.000000000, +0.000000000] | +0.000000000 [+0.000000000, +0.000000000] | 0.920365100 | 0.000000000 |

## Per-checkpoint and per-corpus summaries

| Checkpoint | Scope | Method | Ordered | Gain over same-parent copy [95% CI] | Order gap [95% CI] |
| --- | --- | --- | ---: | --- | --- |
| 271828 | pooled | strict_decoder | 0.915029490 | +0.001000260 [+0.000983304, +0.001017918] | +0.000000808 [+0.000000688, +0.000000930] |
| 271828 | pooled | frozen_probe | 0.908058086 | -0.005971144 [-0.006208470, -0.005749161] | +0.000905028 [+0.000695386, +0.001111592] |
| 271828 | fineweb_edu | strict_decoder | 0.910546126 | +0.001052028 [+0.001019185, +0.001086011] | +0.000000878 [+0.000000629, +0.000001133] |
| 271828 | fineweb_edu | frozen_probe | 0.903004327 | -0.006489772 [-0.006930560, -0.006064427] | +0.000514142 [+0.000161393, +0.000862062] |
| 271828 | fineweb | strict_decoder | 0.909805492 | +0.001066458 [+0.001032372, +0.001101459] | +0.000000722 [+0.000000438, +0.000000999] |
| 271828 | fineweb | frozen_probe | 0.902914296 | -0.005824738 [-0.006272599, -0.005367659] | +0.000489038 [+0.000099849, +0.000872299] |
| 271828 | openwebtext | strict_decoder | 0.915951550 | +0.001017741 [+0.000983850, +0.001053585] | +0.000000775 [+0.000000519, +0.000001044] |
| 271828 | openwebtext | frozen_probe | 0.909321772 | -0.005612037 [-0.006001699, -0.005212025] | +0.000794193 [+0.000470112, +0.001122121] |
| 271828 | wikipedia_en_20231101 | strict_decoder | 0.921736578 | +0.000896717 [+0.000870796, +0.000922784] | +0.000000841 [+0.000000662, +0.000001018] |
| 271828 | wikipedia_en_20231101 | frozen_probe | 0.914903096 | -0.005936765 [-0.006436183, -0.005431299] | +0.001609489 [+0.001163583, +0.002080435] |
| 314159 | pooled | strict_decoder | 0.915119094 | +0.001090823 [+0.001073053, +0.001108734] | +0.000000661 [+0.000000577, +0.000000744] |
| 314159 | pooled | frozen_probe | 0.908155469 | -0.005872802 [-0.006105692, -0.005646086] | +0.000915450 [+0.000705410, +0.001124139] |
| 314159 | fineweb_edu | strict_decoder | 0.910615197 | +0.001168343 [+0.001128106, +0.001210026] | +0.000000668 [+0.000000496, +0.000000842] |
| 314159 | fineweb_edu | frozen_probe | 0.903036479 | -0.006410374 [-0.006857281, -0.005981952] | +0.000524591 [+0.000159336, +0.000867222] |
| 314159 | fineweb | strict_decoder | 0.909969627 | +0.001130346 [+0.001094575, +0.001167855] | +0.000000469 [+0.000000267, +0.000000659] |
| 314159 | fineweb | frozen_probe | 0.902977003 | -0.005862279 [-0.006284603, -0.005412280] | +0.000490988 [+0.000094650, +0.000873776] |
| 314159 | openwebtext | strict_decoder | 0.916241371 | +0.001079656 [+0.001046685, +0.001113865] | +0.000000735 [+0.000000560, +0.000000911] |
| 314159 | openwebtext | frozen_probe | 0.909546402 | -0.005615314 [-0.006004701, -0.005223758] | +0.000796496 [+0.000479696, +0.001123191] |
| 314159 | wikipedia_en_20231101 | strict_decoder | 0.921633261 | +0.001008071 [+0.000981347, +0.001035165] | +0.000000738 [+0.000000596, +0.000000885] |
| 314159 | wikipedia_en_20231101 | frozen_probe | 0.914979699 | -0.005645490 [-0.006144335, -0.005119216] | +0.001632346 [+0.001181965, +0.002114379] |
| 424242 | pooled | strict_decoder | 0.915043462 | +0.001036430 [+0.001016836, +0.001056511] | +0.000000587 [+0.000000507, +0.000000662] |
| 424242 | pooled | frozen_probe | 0.908101500 | -0.005905532 [-0.006138561, -0.005681954] | +0.000904827 [+0.000696702, +0.001114397] |
| 424242 | fineweb_edu | strict_decoder | 0.910717962 | +0.001105630 [+0.001066461, +0.001146303] | +0.000000616 [+0.000000448, +0.000000776] |
| 424242 | fineweb_edu | frozen_probe | 0.903217555 | -0.006394777 [-0.006817769, -0.005978843] | +0.000524397 [+0.000158617, +0.000869878] |
| 424242 | fineweb | strict_decoder | 0.909882142 | +0.001052647 [+0.001013902, +0.001096982] | +0.000000425 [+0.000000236, +0.000000620] |
| 424242 | fineweb | frozen_probe | 0.902912486 | -0.005917009 [-0.006351779, -0.005485559] | +0.000458349 [+0.000066193, +0.000844685] |
| 424242 | openwebtext | strict_decoder | 0.915967451 | +0.001056284 [+0.001016602, +0.001093907] | +0.000000585 [+0.000000416, +0.000000749] |
| 424242 | openwebtext | frozen_probe | 0.909279824 | -0.005631343 [-0.006018166, -0.005240748] | +0.000786009 [+0.000465241, +0.001117246] |
| 424242 | wikipedia_en_20231101 | strict_decoder | 0.921575828 | +0.000953274 [+0.000917566, +0.000989179] | +0.000000683 [+0.000000562, +0.000000808] |
| 424242 | wikipedia_en_20231101 | frozen_probe | 0.914908280 | -0.005714274 [-0.006212489, -0.005202778] | +0.001629186 [+0.001178193, +0.002115140] |
| three_seed_mean | pooled | strict_decoder | 0.915064015 | +0.001042504 [+0.001025339, +0.001060005] | +0.000000686 [+0.000000596, +0.000000772] |
| three_seed_mean | pooled | frozen_probe | 0.908105018 | -0.005916492 [-0.006150681, -0.005693017] | +0.000908435 [+0.000699193, +0.001115978] |
| three_seed_mean | fineweb_edu | strict_decoder | 0.910626428 | +0.001108667 [+0.001073484, +0.001146137] | +0.000000721 [+0.000000534, +0.000000908] |
| three_seed_mean | fineweb_edu | frozen_probe | 0.903086120 | -0.006431641 [-0.006868455, -0.006013955] | +0.000521043 [+0.000159228, +0.000868413] |
| three_seed_mean | fineweb | strict_decoder | 0.909885754 | +0.001083150 [+0.001049114, +0.001119317] | +0.000000539 [+0.000000333, +0.000000749] |
| three_seed_mean | fineweb | frozen_probe | 0.902934595 | -0.005868008 [-0.006301648, -0.005430237] | +0.000479458 [+0.000094080, +0.000864868] |
| three_seed_mean | openwebtext | strict_decoder | 0.916053457 | +0.001051227 [+0.001019153, +0.001084934] | +0.000000698 [+0.000000507, +0.000000885] |
| three_seed_mean | openwebtext | frozen_probe | 0.909382666 | -0.005619565 [-0.006004574, -0.005225367] | +0.000792233 [+0.000470336, +0.001121303] |
| three_seed_mean | wikipedia_en_20231101 | strict_decoder | 0.921648556 | +0.000952687 [+0.000925038, +0.000979837] | +0.000000754 [+0.000000623, +0.000000887] |
| three_seed_mean | wikipedia_en_20231101 | frozen_probe | 0.914930358 | -0.005765510 [-0.006259575, -0.005245785] | +0.001623673 [+0.001178024, +0.002101604] |

## Residual-geometry calibration

Both decoder families add a residual with norm at most 0.25 before normalization, giving angular radius asin(0.25) = 14.477512 degrees.
- Mean fraction of children outside that radius around their learned parent: 0.989823982.
- Mean loose oracle spherical-cap cosine ceiling: 0.982790593.
This is an upper bound that ignores the tighter coordinate box and available information, not an achievable decoder result or evidence that the bound caused the observed performance.

## Historical numerical check

| Seed | Scope | Method | New minus original mean | Within fixed 1e-5 diagnostic tolerance |
| --- | --- | --- | ---: | --- |
| 271828 | fineweb | frozen_probe | +0.000004480864 | True |
| 271828 | fineweb | strict_decoder | +0.000004425775 | True |
| 271828 | fineweb_edu | frozen_probe | -0.000006458707 | True |
| 271828 | fineweb_edu | strict_decoder | -0.000005838782 | True |
| 271828 | openwebtext | frozen_probe | +0.000013040638 | False |
| 271828 | openwebtext | strict_decoder | +0.000011562829 | False |
| 271828 | pooled | frozen_probe | +0.000003793564 | True |
| 271828 | pooled | strict_decoder | +0.000002944401 | True |
| 271828 | wikipedia_en_20231101 | frozen_probe | +0.000004461966 | True |
| 271828 | wikipedia_en_20231101 | strict_decoder | +0.000002244832 | True |
| 314159 | fineweb | frozen_probe | +0.000005913652 | True |
| 314159 | fineweb | strict_decoder | +0.000005780708 | True |
| 314159 | fineweb_edu | frozen_probe | -0.000005896554 | True |
| 314159 | fineweb_edu | strict_decoder | -0.000005313906 | True |
| 314159 | openwebtext | frozen_probe | +0.000012961020 | False |
| 314159 | openwebtext | strict_decoder | +0.000011586983 | False |
| 314159 | pooled | frozen_probe | +0.000004140732 | True |
| 314159 | pooled | strict_decoder | +0.000003379937 | True |
| 314159 | wikipedia_en_20231101 | frozen_probe | +0.000004175480 | True |
| 314159 | wikipedia_en_20231101 | strict_decoder | +0.000002401467 | True |
| 424242 | fineweb | frozen_probe | +0.000004578163 | True |
| 424242 | fineweb | strict_decoder | +0.000004424483 | True |
| 424242 | fineweb_edu | frozen_probe | -0.000006138623 | True |
| 424242 | fineweb_edu | strict_decoder | -0.000005545081 | True |
| 424242 | openwebtext | frozen_probe | +0.000012994686 | False |
| 424242 | openwebtext | strict_decoder | +0.000011668928 | False |
| 424242 | pooled | frozen_probe | +0.000003934801 | True |
| 424242 | pooled | strict_decoder | +0.000003207157 | True |
| 424242 | wikipedia_en_20231101 | frozen_probe | +0.000004281375 | True |
| 424242 | wikipedia_en_20231101 | strict_decoder | +0.000002486927 | True |

## Interpretation limits

- Copy baselines use no fitted decoder. Unweighted pooling maximizes the unweighted average child cosine among single unit vectors copied into every slot; it does not upper-bound distinct slot-specific predictions.
- All reported contrasts use regenerated, matched inputs. Original official results and the historical seed-424242 matched-pooling probe study remain unchanged and separate.
- The released candidate TRAIN probes are not assumed to be identical to the separately fitted matched-pooling probes. The latter three binaries are absent from the release, so no new pooling-probe score or calibrated gain is claimed for them.
- A positive order gap can reflect corpus-wide positional priors; it does not prove recovery of document-specific child order, lossless storage, or preserved mutual information.
- Oracle best assignment uses the target labels to choose a permutation. It is a diagnostic upper envelope, not a deployed decoder or retrieval accuracy.
- These are local four-child reconstructions from direct parents, not decoding all original leaves from unrestricted recursive roots.
- Intervals use 2,000 shared document resamples, conditional on fixed checkpoints/probes. Checkpoint scores are averaged within each document first. There are 606 document units, not 2,424 child units or 24 permutation replicas per document.
