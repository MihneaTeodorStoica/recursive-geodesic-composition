# V2.2 TRAIN-only covariance pilot

No source range has been accessed at this freeze. Pilot seed is 424242; every formulation must use identical fresh initialization, deterministic TRAIN subset/order, optimizer, 250 merger steps, batch size 8, and no decoder stage. The baseline objective remains teacher cosine + existing mean/diagonal-variance/regularizers.

A adds normalized `mean((Cov(g)-Cov(t))^2)` with coefficient 64. B replaces it by the same mismatch after Cholesky whitening using current minibatch teacher covariance plus `1e-4 * trace(C)/D` ridge. C is B computed as an equal mean over the four per-corpus valid terms plus residual-mean squared coefficient .01. Valid covariance requires at least four rows; all terms reject nonfinite values; whitening condition must be <=1e8; gradient norm <=1000; three consecutive saturation steps reject.

Selection is lexicographic: health; cosine within .002 of a corresponding no-covariance baseline; minimum full covariance mismatch; residual mean norm; diagonal variance mismatch. AUC/development/official/downstream are prohibited. A separate cache/preparer must be implemented and frozen before pilot source access; this protocol alone does not authorize it.
