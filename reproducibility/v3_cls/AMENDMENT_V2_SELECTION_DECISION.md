# Amendment V2 development-round-2 selection decision

The predeclared rule selected the late-adversarial protocol:

- Candidate cosine 0.986920 >= 0.985.
- Candidate strongest AUC 0.651691 is 0.043418 below the predeclared round-1 reference 0.695109 (required >=0.03 reduction).
- Decoder child cosine 0.912579 and probe child cosine 0.902638 are within 0.01 of round-1 values.
- Effective rank 121.52 >=100; no nonfinite values or geodesic saturation.

A concurrently evaluated Amendment-V1 baseline on the same round-2 population achieved AUC 0.650813, slightly better than the adversarial candidate. Thus the apparent predeclared improvement is population-dependent, and the direct causal hypothesis that adversarial training improves detectability is **not supported**. Nevertheless, to avoid changing a frozen mechanical selection rule after observing outcomes, V2 is selected exactly as predeclared. This negative finding must be reported. No further objective or hyperparameter candidate will be tested. Seeds 271828 and 314159 will be trained fresh under the exact selected V2 protocol; seed 424242's already-fresh candidate is retained as its final run.
