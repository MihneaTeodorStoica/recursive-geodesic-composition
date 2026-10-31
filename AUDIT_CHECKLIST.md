> Historical supplement checklist. The external-mirror FAIL row refers to the earlier full-repository mirror, not this sanitized source snapshot or a newly verified mirror. PASS rows retain their original evidence scope; current Git-publication checks are recorded author-only outside the branch.

# Final audit checklist

These statuses distinguish the reviewer ZIP from external resources and unestablished reproduction claims. PASS is limited to the evidence cited; it is not proof of perfect anonymity or scientific correctness.

| Check | Status | Evidence |
| --- | --- | --- |
| original scientific source and prior copy | **PASS** | 19 full scientific modules compared byte-for-byte and by SHA-256; INDEPENDENT_SCIENTIFIC_CHECKS.json. |
| historical numeric values and types | **PASS** | 194 historical JSON files and 2,493,640 typed numeric/boolean leaves compared at their nested paths. |
| saved scalar statistics | **PASS** | Independent rank-sum replay of 120 AUCs in 60 files; nine pooled bootstrap mean/interval pairs; INDEPENDENT_STATISTICAL_REPLAY.json. |
| included binary identity | **PASS** | Six probes: complete file SHA-256, size, nested ZIP CRC/member names and complete pickle metadata; INDEPENDENT_PROBE_REVIEW.json. |
| included image metadata and pixels | **PASS** | All five PNG pixel arrays unchanged; no known author metadata; INDEPENDENT_IMAGE_REVIEW.json. |
| known author names handles affiliations paths project urls | **PASS** | Whole-payload bytes/text/SVG/metadata scans and context review found no known author identity. |
| email credentials and secrets | **PASS_SCOPED** | Known author-email and high-confidence GitHub/Hugging Face/AWS/private-key patterns absent; not a formal proof that all possible secrets are absent. |
| project git history ids and saved hostnames | **PASS** | No Git directories/remotes/authors or known project commit IDs; no saved user/host fields. Third-party input revision pins retained. |
| named pdfs and raw identity audit material | **PASS** | No PDF/citation payload or author-only audit/recommendations. Raw identity-bearing evidence remains outside the ZIP. |
| source mirror as anonymous reviewer link | **FAIL_EXTERNAL_NOT_LINKED** | Actual updated 394-file snapshot retains personal log paths and PNG author metadata. It is not linked; see MIRROR_STATUS.md. |
| omitted weights anonymous access | **BLOCKED_EXTERNAL** | 60 omitted weights total 8,062,738,309 bytes; no verified anonymous endpoint. See CHECKPOINT_HOSTING_PLAN.md. |
| full historical from scratch reproduction | **NOT_ESTABLISHED** | Historical source training lineage, caches and older weights are incomplete; strict upstream provenance pins remain. These limits are disclosed. |
| inherited hash discrepancies | **QUALIFIED** | 15 presentation differences, three historical recovery versions and one unresolved fingerprint. Original hash evidence preserved; see HASH_DISCREPANCY_AUDIT.md. |
| fresh extraction commands python compilation formats links | **PASS** | Real candidate extraction; five commands including compileall passed. JSON/YAML/TOML/SVG parsed; zero broken Markdown/evidence paths. EXTRACTED_COMMAND_EVIDENCE.json. |
| real weight placement | **PASS** | Six actual included probes placed and hashed at 12 protocol/lookup paths in a disposable copy. |
| supporting plot regeneration | **PASS** | Two supporting plots generated from saved JSON; zero RGBA pixel differences. |
| perfect anonymity against public search | **NOT_CLAIMED** | Research title, scientific identifiers and artifact hashes remain linkable to public work by deliberate comparison/search. |

Machine-readable checks: `reproducibility/FINAL_PACKAGE_CHECKS.json`. See VALIDATION.md, ANONYMIZATION.md, HASH_DISCREPANCY_AUDIT.md and CHECKPOINT_HOSTING_PLAN.md for methods and limitations.
