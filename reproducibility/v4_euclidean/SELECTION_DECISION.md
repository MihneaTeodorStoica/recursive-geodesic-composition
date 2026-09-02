# V4 architecture selection

Development used the same 4,000 accepted rows and exact shared leaf tensors for all nine checkpoints.

- **Geodesic control:** seeds 424242/271828 were healthy (cosine 0.98698/0.98702; AUC 0.69945/0.69492), but seed 314159 hit the 0.5-radian cap on every evaluated merge, collapsed to effective rank 8.86, cosine 0.86665 and AUC 1.0. The family fails the first numerical-validity gate.
- **Euclidean residual:** all three healthy; mean cosine 0.987009, strongest AUC 0.697732, rank 119.96, decoder/probe cosine 0.91298/0.90312. It is the only family passing every validity and retention gate.
- **Free Euclidean:** mean cosine 0.85389, AUC 0.99979 and rank 49.67; it fails fidelity/AUC/rank gates despite finite normalized outputs.

Therefore **Euclidean residual wins by numerical validity**, not by detectability improvement. Its AUC is essentially the same as the two healthy geodesic controls, so development provides negative evidence that the spherical constraint is the main source of the signature. No protocol revision or additional tuning is allowed.
