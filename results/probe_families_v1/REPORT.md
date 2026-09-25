# Matched probe-family comparison

| Measurement | Mean | Paired 95% CI |
| --- | ---: | --- |
| bounded: learned ordered | 0.912004033 | [0.909822649, 0.914220905] |
| bounded: learned gain | -0.00201747817 | [-0.00221435036, -0.00182612154] |
| bounded: learned delta weighted | -0.0020258123 | [-0.0021681731, -0.00189793457] |
| bounded: learned delta unweighted | -0.00428587315 | [-0.00478812034, -0.00381902401] |
| unbounded: learned ordered | 0.908634124 | [0.906484045, 0.910753591] |
| unbounded: learned gain | -0.0053873865 | [-0.00577235215, -0.00500325548] |
| unbounded: learned delta weighted | 0.000523389007 | [0.000391803637, 0.000648851211] |
| unbounded: learned delta unweighted | -0.00194827165 | [-0.00244395572, -0.0015059154] |
| direct: learned ordered | 0.764288257 | [0.761099219, 0.767398075] |
| direct: learned gain | -0.149733254 | [-0.152644973, -0.146860418] |
| direct: learned delta weighted | -0.000684252078 | [-0.000971438219, -0.000395197393] |
| direct: learned delta unweighted | -0.00405593405 | [-0.0046441118, -0.00346628849] |

All probes are new TRAIN-only fits with equal budgets and schedules. Bounded and unbounded residual probes share zero-last-layer initialization; the non-residual direct MLP starts at TRAIN slot centroids. Scores average three merger checkpoints and three probe initializations within each document; pooling fits are reused across checkpoint pairings. Intervals condition on these fits, not hypothetical training seeds. No validation or test selection is performed.
