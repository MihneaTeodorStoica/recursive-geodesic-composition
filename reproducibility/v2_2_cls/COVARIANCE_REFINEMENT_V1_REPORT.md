# Covariance refinement V1 result

The protocol was frozen in commit `55c2eaf` before execution. Coefficient 64 had weighted covariance/teacher gradient ratio 4.45e-5. The new final ratios rose from .00570 to .37436 without health/saturation failure. Nevertheless the largest fixed-row covariance reduction was only 3.141%, below the predeclared 5% gate. All cosine deltas passed. Therefore no candidate is selected and direct minibatch full-covariance matching is rejected under this setup. No full three-seed run is authorized.
