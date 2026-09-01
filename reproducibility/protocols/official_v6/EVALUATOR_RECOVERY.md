# Official v6 evaluator-only recovery

Official v6 preparation completed and atomically published the declared cache. The first evaluation attempt then failed before loading any safe tensor or producing any metric:

```text
Traceback (most recent call last):
  File ".../evaluate_official_test_v6.py", line 194, in load_cache
    != cfg["confirmation"]["final_results_sha256"]
KeyError: 'confirmation'
```

The evaluator retained two stale v5 provenance-key lookups. V6's frozen config and prepared manifest instead use `development_provenance`, `development_final_results_sha256`, and `post_final_verification_sha256`.

The recovery is evaluator-only. It preserves the original evaluator and spent evaluation marker, verifies their hashes and the exact already-published cache-manifest hash, refuses source preparation or a finalized output, and claims a distinct recovery marker. Its only semantic code change replaces the stale key lookup with checks of the frozen clean-v6 development hashes, implementation parent commit, and no-prior-protected-access flag. It then runs the otherwise identical frozen analysis on the already prepared cache. No source stream is reopened; no checkpoint, attack, bootstrap, threshold, row, or reporting rule changes.

This failure and recovery remain part of the official record regardless of the result.
