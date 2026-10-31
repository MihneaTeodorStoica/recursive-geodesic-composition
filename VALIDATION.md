> The public cached-replay execution evidence below is current. Later checkpoint-publication and packaging sections retain their original audit scope; their no-metric-rerun or failed-old-mirror statements are historical, not verdicts on the separately identified replay. Historical experiment records are not rewritten.

# Validation and preserved historical audit

## Public cached replay: all four CPU workflows PASS

Starting from a clean extraction of the public supplement, the corrected entrypoints were exercised against the pinned, anonymously downloadable checkpoint and cache bytes. Inputs were materialized from prior public downloads, not substituted from a private original cache. This is an explicit `anonymous_cached_replay_v1` execution contract, not a new untouched test collection or satisfaction of the old original-byte provenance guards.

| Workflow | Executed population | Seeds | CPU wall time |
| --- | --- | --- | --- |
| Official | 4,000 paired safe/full documents per seed | 3 | 447.59 s |
| Development | 4,000 paired safe/full documents per seed | 3 | 293.35 s |
| Safe baselines | All 4,000 official safe prefixes | 3 | 879.86 s |
| Recursive reuse | All 606 exact-four prefixes; structural accounting of all 4,000 full documents | 3 | 268.56 s |

The official reporting gate passed. Every seed processed all 9,142,043 available full-document tokens across 11,356 leaves, with zero truncated documents. Official/development/baseline fixed attacks were refitted; applicable statistical summaries used the declared 2,000 bootstrap resamples. Recursive reuse verified the completion/configuration/manifest/result/vector hashes of the newly completed baseline run before its own inference.

Actual result JSON copies are under `reproducibility/replay_validation/`; `reproducibility/ANONYMOUS_REPLAY_VALIDATION.json` records commands, runtime, input boundaries, source-preservation checks and fresh-versus-archived comparisons. Official pooled AUC is `0.5841501666666666`, versus archived `0.5842321666666667`; the difference is disclosed, not used to select or replace an archived result.

The baseline and recursive replay read original official safe tensors, **not** the absent historical regenerated-safe cache. Their paired comparisons record the resulting numerical differences. Recursive full-population accounting uses verified token-count metadata and historical histograms; it is not additional neural inference on every full root. The official replay above separately performs actual full-document inference.

The exercised runtime was Python 3.12.14, PyTorch 2.13.0+cpu, NumPy 2.5.4, scikit-learn 1.9.1 and SciPy 1.18.1, with four CPU compute threads. Timings describe the present machine only; baseline execution overlapped part of official execution and is not an isolated latency benchmark or reproduction of archived GPU timing. The minimal install recipe was not separately installed from scratch. Targeted contract/protocol checks passed **24 tests**; no broad suite is claimed. The documented cache recipe also exited before extraction/continuation on checksum-download and checksum-verification failures.

All 19 scientific modules, 194 historical JSON files and 15 historical configuration files remain byte-identical to the previous public delivery; thus all 2,493,640 retained numeric/boolean leaves are preserved. The 25 reused official scientific helper bodies are AST-identical. Execution/provenance I/O is intentionally changed and fresh outputs use `replays/`, never frozen `results/`. No model/probe training, teacher encoding, raw-corpus download, GPU use or manuscript modification occurred. Full from-scratch training and every other manuscript study are not claimed reproduced.


## Current checkpoint publication: PASS

- Public, ungated model repository: [Hugging Face checkpoint repository](https://huggingface.co/tmlr-embedding-artifacts/recursive-geodesic-checkpoints)
- Immutable Hugging Face revision: `9187157fa4fc6f790be1775475952f14ae33d02e`.
- All 66 original checkpoint sizes/hashes fully acquired and reconciled with authoritative release inventories; complete serialized metadata, ZIP headers/comments/extra fields and CRCs inspected without executing pickle.
- Published 63 exact originals plus three explicitly mapped, length-preserving metadata-only warm derivatives. Static proof preserves all model/optimizer storage bytes, tensor descriptors and non-identifying scientific state. Original experiment hashes remain unchanged.
- Downloaded every hosted checkpoint in full without authentication: **66 files, 8,110,072,711 bytes**, all matching manifest sizes/SHA-256 and actual PyTorch ZIP payloads rather than LFS/Xet pointers. Public card/manifests/checksums and neutral commit authorship checked separately.
- Exercised the real group helper: three final checkpoints downloaded and verified at six protocol/lookup paths; a second invocation downloaded zero files and verified hard-linked aliases.
- Fresh checks preserved all 19 scientific modules, 194 historical JSON files and 2,493,640 typed numeric/boolean values; independently replayed 120 AUCs and nine pooled bootstrap intervals. Maximum errors were `2.220446049250313e-16` and `6.938893903907228e-18` respectively.
- No training, expensive scientific evaluation, teacher/dataset download, cache regeneration or manuscript/result change. A later isolated Python 3.12.14/PyTorch 2.13.0+cpu check safely loaded all 66 files, compared all 191 tensors in each warm pair, strictly loaded 12 final/source/gradient models and exercised synthetic merge/recursive/decoder outputs. Warm pairs produced bitwise-identical outputs; this is not metric replication.

CHECKPOINT_ACCESS.md describes exact commands, groups and original-byte guard limitations. Nine older scaling/matched-pooling payloads remain unavailable outside the 66-file release.

## Recovered caches and partial lineage

Six externally hosted bundles at separate immutable revision `9e4c8e02375f66b3cf9918a20eb81446f996620e` contain 8,075 files, including 8,032 unchanged PT payloads. Original file hashes and 12,049 manifest references matched. Static inspection covered 8,032 complete pickle streams, 645,996 string opcodes, 285,485 ZIP members and 237,293 CRC-checked tensor-storage entries, with no opaque metadata or inspection errors. One identifying official-manifest Git label was neutralized in an explicit derivative, with a matching derivative COMPLETE; all tensors/indexes/sidecars are unchanged. No recognized raw-text/token fields were found, but source identifiers/hashes and embeddings remain linkable.

All recovered tensor files loaded safely on CPU with finite floating tensors. One real row per directory passed the published final model's recursive reducer and strict decoder, with finite outputs and unit-norm parents. Fifteen exact historical source-lineage records passed scoped identity checks; six config hashes match recorded pins. Earlier source weights, trainer/execution bindings and some other study caches remain missing.

`reproducibility/HOSTED_CACHES.json` and `RECOVERED_SOURCE_LINEAGE.json` record these checks and limits. Original official guards reject the metadata derivative and retain additional normalized-layout differences; no guard/result/hash was rebound. Underlying dataset/teacher attribution and terms remain applicable. No historical model-backed metric, training, corpus regeneration or full official audit was rerun.

## Earlier pre-hosting packaging audit

This record describes checks performed for the rebuilt review package. The prior package's claims were treated as hypotheses and checked against the original read-only source, previous ZIP, actual live mirror downloads and current release inventory. Historical experiment audits are not represented as newly rerun model evaluations.

## Scientific integrity: PASS

- Independently CRC-checked/extracted the previous supplement and supplied mirror archive. The previous ZIP SHA-256 was `288e1a2de555e6199774bab3ad59c96869f6e2d669419ea1a6daa04a6bf1dcc7`.
- All **19 scientific modules** under `src/recursive_geodesic_composition/` match the original repository and prior archive byte-for-byte. The current served source modules also match. No method/RNG/loss/mask/training/model changes were introduced.
- All **194 retained historical JSON files** were compared by nested numeric key/index path: **2,493,640 numeric/boolean leaves unchanged** across original, previous package and latest audited mirror. All original `src/`, `configs/`, `scripts/` and `results/` files included in the prior package remain byte-identical to that package in this rebuild.
- All five included PNGs have exactly the same dimensions/RGBA pixels as their original and previous counterparts. Identifying ancillary metadata had already been removed in the prior package and is not reintroduced.
- Table numbering/captions 4–17 were checked against the current original anonymous 40-page submission PDF; literal evidence-map paths were verified. The paper remains unchanged and is not included/rebuilt in this ZIP.

File-level numeric/source/image evidence is retained in `reproducibility/INDEPENDENT_SCIENTIFIC_CHECKS.json` and `INDEPENDENT_IMAGE_REVIEW.json`. Preservation of source bytes/numeric records is not independent proof that every reported scientific result is correct.

## Saved statistics: PASS; no model rerun

The prior helper's `report` command was rerun successfully, but that alone was not accepted as independent validation. A separate rank-sum ROC-AUC implementation, independent of the supplied helper's pair-count algorithm, recalculated **120 AUC points from 60 saved classifier-prediction files**. Maximum absolute difference from the saved raw AUCs: `2.220446049250313e-16`.

Five pooled teacher-fidelity means were separately recomputed from the 4,000 saved document rows. The population contains 2,707 two-child, 687 three-child and 606 four-child cases. The learned-parent mean is `0.9850454794397708`; token-weighted pooling is `0.9829381574051947`.

The unchanged NumPy-only `shared_bootstrap_means`, `summarize_scores` and `fidelity_summary` function bodies were extracted from the original scientific module and executed without importing Torch. All **nine means and nine 95% intervals** were replayed on the saved scalar scores with **2,000** resamples and seed `22092601`; maximum absolute discrepancy was `6.938893903907228e-18`. Details are in `reproducibility/INDEPENDENT_STATISTICAL_REPLAY.json`.

These are scalar-statistic checks, not reconstruction of raw-vector cosines, original official prediction arrays, fresh classifier fits, teacher encoding, GPU inference, latency benchmarks or training. `supplement.py report` itself replays means and AUC points, not the intervals. Its historical official AUC output is a copied aggregate, not an independently recovered original official bootstrap.

## Checkpoint identities and metadata: scoped PASS

- Cross-checked all **66** `.pt` manifest entries by exact filename, byte size and SHA-256 against the current release-asset inventory.
- Fully rehashed all **six included historical probes** (47,334,402 bytes). Their sizes/hashes match the original immutable manifests. Checked nested ZIP CRCs/member names, complete pickle metadata strings and full-byte known-identity patterns without unpickling/loading tensors. No known author identity was found.
- Independently fetched complete leading `data.pkl` opcode streams for all 66 external/or included release files through HTTP byte ranges, without unpickling or tensor loading. Found the known personal-path issue in all three external warm checkpoints. These are excluded and flagged; no derivative was created.
- The other 60 full payloads were **not** fully downloaded or locally rehashed. Matching release digests and inspected leading metadata is not full local integrity/anonymity certification of omitted tensor bytes.

See `reproducibility/INDEPENDENT_PROBE_REVIEW.json`, `INDEPENDENT_CHECKPOINT_METADATA_REVIEW.json`, `WEIGHTS_MANIFEST.json` and `CHECKPOINT_COLLECTIONS.json`.

## Historical source-hash investigation: qualified

All 19 previously reported existing-path mismatches were independently checked. Exact expected content was found for six of seven unique target fingerprints in reachable source objects or preserved recovery files. The detailed per-record classification is:

| Classification | References | Meaning |
| --- | ---: | --- |
| Presentation/serialization difference | 15 | Exact historical source recovered; differences are table caption/label placement or descriptive presentation, not score calculations. |
| Expected historical version difference | 3 | Documented pre-score evaluator or pre-encoding population-recovery versions, preserved separately. |
| Genuine artifact discrepancy established | 0 | None established among these 19 references; not a blanket scientific-result certification. |
| Unresolved | 1 | Exact failed initial `prepare_safe_baselines.py` bytes were not recovered; its fingerprint remains unchanged and unresolved. |

Five exact identity-free historical source/configuration copies are included as **non-active provenance references**. The historical renderer containing personal metadata is not reintroduced at its exact original bytes. No original source-hash mismatch was silently rebound to current code. Strict readers can still reject old source fingerprints and fresh chained runs require missing tensors and an explicitly frozen current output chain. See HASH_DISCREPANCY_AUDIT.md and its machine-readable counterpart.

## Live mirror: FAIL, independently observed

The actual unauthenticated reviewer surface and complete live ZIPs were inspected, not just simulated. The initial 394-file live download matched every supplied mirror-ZIP payload. After the authors' updates, citation/BibTeX names and institution redaction were verified fixed. The latest fully checked snapshot still exposed personal log paths and PNG author metadata; rewritten release/preprint URLs failed as nonexistent repository files. A temporary rate-limit pause was followed by a successful refreshed download.

The failed mirror is intentionally not linked in this reviewer package. MIRROR_STATUS.md records the neutral decision. Detailed raw identity-bearing evidence and repair expressions are author-only deliverables outside the ZIP. This conclusion is snapshot-specific; a future repaired mirror needs another logged-out audit.

## Fresh extraction, commands and file formats

The final archive is subjected to ZIP CRC/path/member-metadata checks, extraction into a new directory, package-manifest and SHA256SUMS verification, Python compilation, JSON/YAML/TOML/SVG parsing, local-documentation path checks, and the actual README helper commands. A disposable extracted copy exercises real included-probe placement without changing the archive's evidence copy. Supporting plot generation is run from saved JSON in a separate workspace and compared at the pixel level.

Exact final statuses/counts are recorded in `reproducibility/FINAL_PACKAGE_CHECKS.json`. The external delivery report holds the final ZIP size and digest; no self-referential ZIP hash is embedded here. Deterministic packaging uses fixed timestamps/permissions, relative safe regular-file paths, no ZIP comments/extra fields, no symlinks and no bytecode caches. Package-level hashes are regenerated for this rebuild; older audit/checksum scopes remain historical.

## Final anonymity method

The final extracted payload is checked for known author names/variants/handles/emails/affiliation, personal paths, original project-owner URLs, project Git identifiers and high-confidence credential formats. Text/Markdown/source/manifests, SVG/XML, PNG metadata and nested probe metadata are inspected separately. Named PDFs, raw identity-bearing audit documents and Git history/remotes are not included. A string-scan pass alone is not described as proof of anonymity. Necessary public scientific titles, hashes and third-party revisions remain linkable by deliberate search/comparison.

## Environment and checks not performed

Packaging host: Python 3.14.8, NumPy 2.5.3, PyYAML 6.0.3, Matplotlib 3.11.2 and Pillow 12.3.0 on Linux x86-64. The project's documented historical environment remains Python 3.12.14/PyTorch 2.13.0+cu130/NumPy 2.5.1/CUDA; it was not installed or reproduced for this rebuild. Installation/data/model recipes were inspected, not executed.

No retraining, continuation update, probe fitting, GPU inference, teacher/dataset download, source-corpus scan, cache generation, expensive experiment, full model-backed historical audit or broad test suite was run. No anonymous host, account, upload, visibility/settings change, manuscript modification or submission action was performed. Full from-scratch source-model reproducibility and anonymous access to the omitted model collection remain unavailable.
