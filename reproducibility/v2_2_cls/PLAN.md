# V2.2 CLS plan — pre-TRAIN freeze

## Allocation proof

TRAIN `[244000,256000)`, development `[256000,265000)`, buffer `[265000,266000)`, official `[266000,296000)`. `scripts/v2_2_cls/protocol.py` declares all historical v1/V2.1 intervals and refuses any overlap. Only TRAIN is authorized initially; development/official are prohibited until their respective freezes.

## Hypothesis and fixed candidate pilot

V2.1 diagnostics found covariance/correlation mismatch and Wikipedia amplification despite high CLS cosine. Pilot candidates (TRAIN only; no sweep): (A) normalized covariance Frobenius; (B) teacher-whitened covariance; (C) A plus equal per-corpus averaging; (D) C plus a small residual-mean penalty. Select once using held-out TRAIN criteria: finite condition/gradient behavior, lower covariance Frobenius, no teacher-cosine/reconstruction degradation. The selected formulation, coefficient, stages, and all three fresh random seeds must then be frozen before fresh training.

## Invariants

Same CLS teacher revision, K=4 chunking, architecture, staged structure, teacher/reconstruction objectives, attacks and bootstrap. Covariance loss must use centered batch covariance, stable ridge/normalization, finite and condition-number assertions, and logged gradient norms. No geodesic/salience change absent TRAIN evidence. V2.1 official is never loaded for selection.

## Acceptance criteria

Development success requires lower pooled and per-corpus AUC (especially Wikipedia), reduced covariance mismatch, teacher cosine in V2.1's high regime, and no collapse, saturation, or retention failure. At most two documented protocol revisions; each requires fresh TRAIN and fresh non-reused development allocation. Official remains one-shot after complete evaluator/probe freeze.
