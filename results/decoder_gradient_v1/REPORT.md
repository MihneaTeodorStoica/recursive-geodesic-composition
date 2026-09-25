# Decoder-parent gradient intervention

| Measurement | Mean | Paired 95% CI |
| --- | ---: | --- |
| Teacher: attached minus detached | -1.56530951e-06 | [-1.99045976e-06, -1.14239122e-06] |
| AUC: attached minus detached | -0.000426833333 | [-0.000755447917, -0.000109397917] |
| ordered delta | -8.36244596e-05 | [-8.51631546e-05, -8.20019859e-05] |
| gain delta | 5.54807957e-06 | [5.49096143e-06, 5.60837447e-06] |
| order gap delta | -7.09280848e-08 | [-9.01911907e-08, -5.24946295e-08] |

New 640-document exact-four TRAIN regime. Each checkpoint pair shares the identical 300-update warm model, optimizer, RNG and all 100 joint minibatches. The only joint-stage switch is decoder-parent detachment; original teacher, distribution and encoder regularizers remain active. Both arms continue fitting the decoder. This is not a recreation of historical full-tree clean-v6 training, not training from scratch, and not removal of all decoder computations. Fixed final step, no validation selection.
