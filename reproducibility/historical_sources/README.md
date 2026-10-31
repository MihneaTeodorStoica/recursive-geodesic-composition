# Recovered historical source snapshots

Every included file is an exact historical byte copy; its SHA-256 is also the neutral directory key. These are forensic provenance inputs only: they are **not active code/configuration**, are not supported entry points, and must not replace current source files or current checksum bindings. No project commit identifiers are included.

| Directory | Historical recorded path | Status |
| --- | --- | --- |
| `sha256-7c5d6ff432c27be67d0ee2e8d9cdc2cff476d733a773b5c1f11d4041e3c9ed73/` | `scripts/evaluate_safe_baselines.py` | Exact bytes recovered from source Git history; change from current source is TeX caption/label placement only. |
| `sha256-67ca34f54a7c0dd90d8ead4f942be91ce9032a00dc6fc9e69e833f77f871d550/` | `scripts/render_continuation_comparison.py` | Exact bytes recovered from source Git history; changes are rendered-table caption/label presentation. |
| `sha256-adbd9258986d32b70b25d32edd327b348290417765f75109ae53e62796fe33db/` | `scripts/render_reconstruction_calibration.py` | Exact bytes recovered from source Git history; changes are rendered-table caption/label placement. |
| `sha256-283180e037ddc92c3c0e9b622c9c73bb90aaec197a5d389de5da0679a2ee5aa8/` | `scripts/evaluate_longembed_review.py` (historical failure) | Exact bytes preserved in the recovery archive as `evaluate_longembed_pre_scoring.py`; no retrieval scores existed before its numerical repair. |
| `sha256-cb3b170d78d7f801f34a1db7c45a4c1440de3ea918086cf8c701ba0283645381/` | `configs/review_completion_v1.yaml` (pre-recovery) | Exact bytes preserved as `PRE_RECOVERY_CONFIG.yaml`; superseded before fitting/evaluation after the fixed-range population check. |

Two further digests were examined but deliberately not copied here. The exact historical `scripts/render_recursive_reuse.py` bytes are present in Git history, but the file contains identifying image metadata and is not safe to add as an exact reviewer-facing copy. The expected historical `scripts/prepare_safe_baselines.py` bytes were not found in the reachable Git objects or the inspected archive, so no substitute is supplied. See [the hash audit](../../HASH_DISCREPANCY_AUDIT.md) for classifications, consumers and limitations.
