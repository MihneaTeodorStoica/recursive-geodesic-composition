# Preserved pre-run test dependency failure

The first full-suite invocation stopped before cache preparation or training with 200 passes and one failure. The failing legacy storage-verification test requires `results_geodesic_tangent_angle05_equal_encoder_1500/encoder_seed424242.pt`, an old checkpoint deliberately deleted for storage. This is unrelated to v6 code and is consistent with the declared missing original encoder checkpoints.

The failing log remains append-only at `results_kary4_strict_autoencoder_v6_clean_development.FULL_SUITE.log`. Recovery does not recreate or access any source data: rerun the suite while deselecting exactly that unavailable-artifact assertion, preserve the second log separately, run all static checks, then continue the frozen pipeline. No v6 namespace is overwritten.
