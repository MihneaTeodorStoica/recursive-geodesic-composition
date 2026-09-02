# V2.2 TRAIN-only covariance refinement V1

Original pilot remains final historical evidence; formulation A is not promoted because its fixed-row covariance improvement was only .000503%. Replay at A's final TRAIN checkpoint on one deterministic balanced 8-row batch found weighted gradients: teacher .147322, mean .057617, diagonal variance .072100, covariance coefficient 64 only 6.556e-6. Thus covariance/teacher gradient ratio was 4.45e-5; coefficient 64 was effectively inactive.

This frozen refinement uses BASELINE_R and four logarithmic full-covariance coefficients 8192, 32768, 131072, 524288. The existing dimension-normalized mean squared covariance-element difference is unchanged; whitening is prohibited. No component-specific clipping is added because even the largest estimated initial covariance gradient remains below the frozen global health ceiling; the existing optimizer/update path and gradient-health threshold 1000 apply identically. Seed 424242, cache, deterministic rows/order, batch 8, 250 merger-only updates are shared.

Eligibility: finite/no persistent saturation/gradient <=1000; fixed-row cosine >= BASELINE_R-.002; covariance mismatch reduction >=5%. Winner is maximum reduction. Otherwise reject direct minibatch covariance matching.
