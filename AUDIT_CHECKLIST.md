# Review artifact audit checklist

Current checkpoint evidence is in `reproducibility/CHECKPOINT_PUBLICATION_VALIDATION.json` and CHECKPOINT_ACCESS.md. The older `FINAL_PACKAGE_CHECKS.json`, literal command transcripts and pre-hosting validation sections are explicitly historical; their old missing-weight/failed-mirror rows are not current availability verdicts.

| Check | Status | Evidence and boundary |
| --- | --- | --- |
| Scientific source | **PASS** | All 19 scientific modules unchanged; no model/RNG/loss/training changes. |
| Historical numerical evidence | **PASS** | 194 historical JSON files; 2,493,640 typed numeric/boolean leaves unchanged. Tables 4–17 keep their evidence mapping. |
| Saved scalar statistics | **PASS** | Independently replayed 120 AUCs and nine pooled bootstrap mean/interval pairs; no neural rerun. |
| Included historical probes | **PASS** | All six exact originals remain unchanged, totaling 47,334,402 bytes. |
| Public model release | **PASS** | 66 artifacts at immutable HF revision: 63 originals and three declared metadata-only derivatives; public and ungated. |
| Anonymous full checkpoint downloads | **PASS** | Every checkpoint downloaded fully without authentication; 8,110,072,711 bytes matched sizes/SHA-256; actual ZIP payloads, not pointers. |
| Full serialized metadata and warm-state preservation | **PASS_SCOPED** | Complete metadata/archive inspection without pickle execution. Warm storage bytes, tensor descriptors, optimizer/numeric state and non-identifying strings unchanged. No dynamic Torch load/forward claimed. |
| Pinned group download helper | **PASS** | Real final-group download: three models, six protocol/lookup paths, hard-linked aliases; second invocation downloaded zero files. |
| Python and structured formats | **PASS** | Python compiled and parsed for Python 3.12 grammar; JSON/YAML/TOML/SVG parsed. Package-manifest/checksum verification applies to the delivered snapshot. |
| Known identity, credentials and original Git identifiers | **PASS_SCOPED** | Whole-tree scans, binary/static metadata inspection and public card/commit inspection; raw identifying audit materials stay outside public artifacts. No claim that every conceivable secret is detectable. |
| Source mirror | **SNAPSHOT_SPECIFIC** | Anonymous source URL is in README/MIRROR_STATUS. Actual complete downloaded snapshots are compared byte-for-byte; auto-update is not an immutable source pin. Final delivery reports identify the observed snapshot privately. |
| Older payloads and caches | **LIMITED** | Nine older scaling/matched-pooling binaries remain unavailable outside the released 66; teacher/data/vector caches are not supplied. |
| Historical source hashes | **QUALIFIED** | 15 presentation differences, three historical recovery versions and one unresolved fingerprint preserved; no old result hash is rebound. |
| Full historical from-scratch reproduction | **NOT_ESTABLISHED** | Final clean-v6 is 300+100 continuation from exact sources; earlier source-training lineage remains incomplete. |
| Perfect anonymity | **NOT_CLAIMED** | Public title/results/hashes remain searchable; platform operators may know uploader/source identity. |

The supplementary ZIP stays below 100 MB and excludes multi-gigabyte hosted weights and author-only audit reports. Its checksum is delivered outside the archive to avoid self-reference. Reproduction prerequisites and original-byte warm guard limitations remain explicit in REPRODUCING.md and CHECKPOINT_ACCESS.md.
