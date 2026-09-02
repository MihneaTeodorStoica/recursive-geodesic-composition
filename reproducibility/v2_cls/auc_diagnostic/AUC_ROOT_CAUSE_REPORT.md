# V2.1 official AUC diagnostic

## Observed facts

This diagnostic used only the frozen official safe cache and frozen checkpoints. Across seeds, every corpus is detectable by the established attack, but Wikipedia is consistently the dominant contributor: per-corpus logistic diagnostic AUC is about .75--.77 for Wikipedia, versus .62--.66 for FineWeb-Edu, FineWeb, and OpenWebText. Thus detectability is broad, with a strong Wikipedia amplification.

Teacher cosine is .9868--.9869 pooled, while average residual norm is about .150. The residual has a nonzero global mean (norm .0119--.0125), but its mean alignment with individual residuals is only .078--.081. This is evidence against a single global residual direction explaining most per-document error.

Coordinate diagonal variance differences are small (.00027--.00036 norm pooled), while covariance Frobenius differences are materially nonzero (.00482--.00493 pooled; .0060--.0077 within corpus). Wikipedia has the largest mean shift, diagonal-variance mismatch, covariance mismatch, and worst cosine. This supports covariance/correlation and domain interaction as likely contributors beyond first moments.

The learned correction angle is small (mean .031--.034 rad, maximum .065--.067) and is not by itself evidence that correction causes separation. The diagnostic has not established causal attribution between correction and attack score.

## Strong evidence

1. The effect is not single-corpus-only, but Wikipedia is the largest observed contributor.
2. A small systematic mean shift exists, yet residual directions are weakly aligned; separation is likely distributed rather than a single universal translation.
3. Covariance mismatch persists despite high paired cosine and is a plausible exploitable feature.

## Unresolved

Development-vs-official source-shift and PCA/control-ablation conclusions were not computed in this limited diagnostic pass, so no causal explanation of the development/official difference is claimed. RBF versus logistic, bootstrap CIs, and internal-score correlations should be added only as a separate immutable diagnostic expansion.

## TRAIN-only V2.2 candidates (hypotheses, not tuned on official)

1. Add full covariance/whitened-space distribution matching on TRAIN.
2. Add a TRAIN residual mean-bias penalty, evaluated with held-out TRAIN/development only.
3. Add corpus-balanced distribution losses / recursive generated-child exposure, with predeclared TRAIN-only ablations.
