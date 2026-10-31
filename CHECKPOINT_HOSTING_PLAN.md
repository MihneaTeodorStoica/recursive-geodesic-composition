# Checkpoint hosting plan

## Scope and status

This is a conditional, reviewer-safe hosting plan, not evidence that a mirror has been created, that a host has accepted the files, or that a reviewer has accessed them. No account was created, no file was uploaded, and no external service was modified for this audit. There is no working host URL or retrieval command to give. Do not add a guessed URL or point a download command at an uncreated record.

The inventory was cross-read from `reproducibility/ARTIFACT_MANIFEST.json` (12 historical weights), `reproducibility/REVIEW_RELEASE_MANIFEST.json` (54 new control weights), and `reproducibility/WEIGHTS_MANIFEST.json` (all 66 weights). Names, protocol destinations, roles, sizes and SHA-256 identities are retained exactly. Counts and byte sums are computed from those fields. This rebuild independently fully rehashed the six included probes and matched all 66 release-asset sizes/digests; it did not fully download or locally rehash the other 60 payloads. See VALIDATION.md. Do not describe omitted-file manifest identities as fresh full-payload verification.

## Weight inventory: all 66 `.pt` assets

The six historical matched parent-only probes are already packaged in `weights/`; the other 60 files are external. Sizes below are exact decimal bytes.

| Role | Exact names / expansion | Count | Bytes per file | Total bytes | Status and function |
|---|---|---:|---:|---:|---|
| Final clean-v6 model | `autoencoder_seed424242.pt` (494,519,415); `autoencoder_seed271828.pt`, `autoencoder_seed314159.pt` (494,519,479 each) | 3 | as listed | 1,483,558,373 | External. Required for three-seed final-model inference/evaluation. One seed alone supports only a single-checkpoint inference, not the reported three-seed aggregate. |
| Continuation-initialization source model | `source_seed424242.pt` (494,517,367); `source_seed271828.pt`, `source_seed314159.pt` (494,517,431 each) | 3 | as listed | 1,483,552,229 | External. Required to rerun the seed-matched clean-v6 continuation and for source-versus-final comparisons and the official pre-access checks. Does not by itself recreate the earlier source-model training lineage. |
| Historical candidate parent-only probes | `candidate_parent_only_probe_seed{424242,271828,314159}.pt` | 3 | 7,889,085 | 23,667,255 | Included in the supplement. Needed for the corresponding fixed-probe candidate analyses; not model checkpoints. |
| Historical source parent-only probes | `source_parent_only_probe_seed{424242,271828,314159}.pt` | 3 | 7,889,049 | 23,667,147 | Included in the supplement. Needed for source-side fixed-probe comparisons; not model checkpoints. |
| Attached/detached intervention models | `review_attached_seed{271828,314159,424242}.pt` and `review_detached_seed{271828,314159,424242}.pt` | 6 | 494,521,397 | 2,967,128,382 | External. New, additive intervention model outputs; not substitutes for the original final clean-v6 models. Needed to replay the model-backed results of this particular intervention without retraining. |
| Shared intervention warm checkpoints | `review_warm_seed{271828,314159,424242}.pt` | 3 | 591,173,580 | 1,773,520,740 | External in the original inventory, but **unsafe to redistribute unchanged**: the serialized metadata contains a personal filesystem path. These are intermediate 300-update warm states, not final inference models. The existing `scripts/evaluate_decoder_gradient.py` creates each warm state from its source checkpoint, then forks attached/detached arms; the warm files are therefore not required for the existing full rerun path. |
| New probe-family weights | For each `family ∈ {direct,bounded,unbounded}`, `pseed ∈ {731241,731242,731243}`, and `variant ∈ {learned_271828,learned_314159,learned_424242,unweighted,weighted}`, the exact filename is `review_{family}_{pseed}_{variant}.pt` | 45 | See next table | 354,978,585 | External. New TRAIN-only fits for the additive probe-family comparison; optional to the main results, required to replay this specific fitted-probe control from its frozen weights. |

The 45 new probe files have this exact per-family breakdown for **each** of the three `pseed` values:

| Family | Three `learned_*` files (each) | `unweighted` | `weighted` | Bytes per `pseed` group | Three-group total |
|---|---:|---:|---:|---:|---:|
| `direct` | 7,888,421 | 7,888,373 | 7,888,349 | 39,441,985 | 118,325,955 |
| `bounded` | 7,888,433 | 7,888,385 | 7,888,361 | 39,442,045 | 118,326,135 |
| `unbounded` | 7,888,457 | 7,888,409 | 7,888,385 | 39,442,165 | 118,326,495 |

For example, the expansion for `direct` includes `review_direct_731241_learned_271828.pt`, `review_direct_731241_learned_314159.pt`, `review_direct_731241_learned_424242.pt`, `review_direct_731241_unweighted.pt`, and `review_direct_731241_weighted.pt`, then the same five suffixes for `731242` and `731243`. Apply the same exact expansion for `bounded` and `unbounded`. This is 3 families × 3 probe seeds × 5 representations = 45 files. The byte totals include all 45, not just the learned-seed fits.

**Reconciled totals**

- Six included historical probes: **47,334,402 bytes**.
- Sixty external files as originally manifested: **8,062,738,309 bytes** (six model checkpoints + 54 new review-control weights).
- All 66 `.pt` files: **8,110,072,711 bytes**.
- The 54 weights listed in `REVIEW_RELEASE_MANIFEST.json` total **5,095,627,707 bytes**: six attached/detached models, three warm checkpoints, and 45 new probes.
- Warm files alone total **1,773,520,740 bytes**; a sanitized reserialization will have a new size and hash, which have not been measured.

## Minimum weight tiers by goal

“Minimum” here means the weights needed for the specified model-backed task. It does not supply omitted teacher/dataset caches, task data, or GPU execution, and it does not convert saved scores into a fresh run.

| Goal | Minimum exact weight set | Exact weight bytes | What remains outside the weight tier |
|---|---|---:|---|
| Read/replay included scalar results with the package helper | None | 0 | `python supplement.py report` re-aggregates saved scores; it does not run a model, teacher encoding, training, timing, or bootstrap replay. |
| One seed's inference on already-computed leaves | `autoencoder_seed424242.pt` | 494,519,415 | Ordered teacher leaves and exact positive token masses must already exist. This is a one-seed result only. |
| Three-seed final-model inference; tree/resegmentation and LongEmbed controls | All three `autoencoder_seed*.pt` files | 1,483,558,373 | Each control also needs its pinned input data/cache and appropriate teacher/setup. These tasks do not need the three continuation-source checkpoints. |
| Input-order control | Three final `autoencoder_seed*.pt` files plus the three included `candidate_parent_only_probe_seed*.pt` files | 1,507,225,628 total; **1,483,558,373 external** | `scripts/evaluate_input_order.py` loads the three candidate probes, not the source probes; the control also needs the saved 606-case inputs and protocol. |
| Main source-versus-final comparisons and official-v6 pre-access/evaluation | Three final models + three source models + all six included historical parent-only probes | **3,014,445,004 total; 2,967,110,602 external** | Official evaluation still needs the pinned official cache/teacher/corpora. The original README says the official pre-access path checks the source models as well as the final models and six probes. |
| Rerun clean-v6 continuation training | Three source models | 1,483,552,229 | Recreate the pinned TRAIN cache and use the declared configs/environment/GPU; the resulting final models and probes are outputs. This is continuation, not from-scratch recovery of the older source-model training. |
| Replay the fitted new probe-family control from released fits | Three final models + all 45 probe-family files | **1,838,536,958 external** | Recreate the exact TRAIN/evaluation cache and teacher/dataset inputs. The evaluator also fits probes; a fresh fit is a new execution, not proof of byte-identical historical weights. |
| Replay the decoder-parent gradient intervention using its frozen final models | All six `review_attached_*` and `review_detached_*` files | 2,967,128,382 | Inputs/caches and the control protocol remain required. Warm intermediates are not needed to load these six final models. |
| Rerun the gradient-intervention training protocol from its declared source | Three `source_seed*.pt` models | 1,483,552,229 | `scripts/evaluate_decoder_gradient.py` regenerates three warm states from those sources, then trains both joint-stage arms; the TRAIN cache, environment, and GPU are also required. The six intervention models and warm files are then outputs. |

The core main-evaluation tier consists of **12 files**. Six small probes (47,334,402 bytes) are already in the ZIP; its external prerequisite is the six model checkpoints (2,967,110,602 bytes). Even this external model-only set exceeds the TMLR supplement upload limit by nearly 30×. The whole 66-file weight inventory is about 81.1× the 100,000,000-byte supplement limit. Do not label an archive-only scalar-score check as a model-backed reproduction.

## Missing historical scaling and matched-pooling weights

The 66-file inventory is not a complete inventory of every older experiment's binary artifacts. In particular, do not silently substitute clean-v6 or the 45 new control probes for these separate inputs.

- The historical matched-pooling run records three separately fitted probe identities in `results/v1_k4_matched_pooling/PROBE_MANIFEST.json`: `results/v1_k4_matched_pooling/probes/mean_parent_only.pt` (SHA-256 `1b893f5f95d56d2c2f260705ad2deebb7e73f9b56c1db95a0e4f7a12d1cbdef1`), `.../v1_parent_only.pt` (`4081573c2c42a371c617bce8da1e5c438a1a2d29e03d5102847b089149d05839`), and `.../weighted_parent_only.pt` (`7f2b18a267f2d3b6f1833fc3bf98a529b2a268268677910e97e29ddf362ab488`). These paths and hashes are recorded, but the three `.pt` payloads and their byte sizes are not in the release manifests or prior ZIP. The analysis also used the already-manifested `autoencoder_seed424242.pt`; the missing probes are distinct from the six included source/candidate probes.
- `results/kary_scaling/ARTIFACT_MANIFEST.json` refers to three separate model checkpoints—`results_kary4_leaf510_vs1024_v1_leaf1024/encoder_seed424242.pt`, `results_kary8_leaf768_validation_v1/encoder_seed424242.pt`, and `results_geodesic_tangent_angle05_equal_encoder_1500/encoder_seed424242.pt`—and three separate retention probes—`results_kary8_leaf768_validation_v1/retention_probes/parent_only_kary4_reference.pt`, `.../parent_only_kary8.pt`, and `.../parent_only_binary_reference.pt`. Their manifest hashes are in that file; it gives no byte sizes. None is among the 66 listed weights or either release manifest. The corresponding scaling configuration is validation-only (`configs/kary_scaling.yaml`, `configs/kary4_reference.yaml`); the prior evidence map says the tree does not supply these model/probe payloads or the matched validation cache. A numerical rerun requires recovering/recreating those exact validation models and cache, not downloading a final clean-v6 checkpoint.

These absences affect numerical reruns, not offline inspection of the already tracked report/table values. Do not fabricate byte counts, claim these old assets are hosted, or describe regenerated models as the historical checkpoint bytes.

## Recommended anonymous host, conditional on author-side verification

### Primary candidate: one restricted Zenodo record/link

Zenodo's current docs specify up to **50 GB and 100 files per record**; even all 66 original `.pt` payloads total 8,110,072,711 bytes and fit the byte/file ceilings, and the 60 external files plus a manifest/checksum/readme would be fewer than 100 files. Each model checkpoint is under 600 MB. The matching author-neutral staging layout should be a flat `weights/` directory using the exact `asset_filename` values from `reproducibility/WEIGHTS_MANIFEST.json`; keeping filenames flat matches the existing local materializer. Upload only the 57 non-warm external files at original bytes by default; the six historical probes remain in the supplement. Do not upload the three warm files unchanged.

For the warm role, prefer leaving the binary out: the current intervention script regenerates it from source checkpoints. If a branch-resume copy is essential, work only in a clean, controlled copy: first verify the original hash; load with a restricted/weights-only loader if supported; replace only identity-bearing metadata strings/path fields; preserve and compare every model tensor, optimizer state, CPU/CUDA RNG state, schedule, and training configuration value; save a new binary; then compute a new file SHA-256/size and a per-state comparison record. This is a different serialized artifact. Keep both source and derivative hashes in an author-only provenance record, but expose only the derivative hash to reviewers. Do not claim original binary identity. The existing `supplement.py materialize` checks the original warm hash and will reject such a derivative; the current evaluator has no separate warm-resume entrypoint. Thus a sanitized warm file must be handled by a separately declared derivative manifest and a compatible local resume path, or omitted. No sanitization has been performed.

Zenodo supports secret-link permissions including `Can preview draft` and `Can view` for restricted files, and optional expiry. A draft-preview link is a plausible review-only route; however, the documentation does not establish that this specific link is accessible without a Zenodo login or that the uploader identity is hidden from the viewer. Zenodo also requires a creator field, and published-record metadata is public even when files are restricted. Therefore: use a draft with restricted files and a short-lived preview link only if truthful neutral metadata can be used and a logged-out/clean-browser test confirms both access and absence of identity disclosure. Do not publish a DOI or public record during review merely to obtain capacity. If either identity or no-login access fails, do not use Zenodo for this double-blind link.

### Secondary candidate: a neutral public model-hosting namespace

Hugging Face Hub supports large model files and public repositories with no per-repository size cap; its docs recommend files below 200 GB, so these sub-600-MB checkpoints are technically modest per file. But free public storage is explicitly best-effort, with a caution that beyond the first few GB users should use storage responsibly; paid tiers and grants have different limits. The 8.1-GB full inventory is not guaranteed by a free account. The Hub has no anonymous-review redaction mode: a public namespace, card, commit history, account/org details, and file metadata must be checked for identity leaks. Use this only if the authors can confirm sufficient quota and an account boundary that does not reveal the authors, and then verify all links from a logged-out browser. No Hub repository was created or capacity tested.

### Not suitable for these binaries: text-redaction mirrors

The current Anonymous GitHub FAQ describes one-per-line regular-expression custom redactions for text terms, plus automatic owner/organization/repository-name redactions. It also says only text-based files are anonymized, binaries are served as-is, files larger than 8 MB are unsupported, and replacements in source can change program behavior. The FAQ mentions file filters but does not establish binary redaction or provide a dependable exclusion guarantee for these checkpoint files. The 0.49–0.59-GB models/warm checkpoints exceed the per-file limit; smaller probes remain binary and therefore are not metadata-sanitized by that service. It is not a checkpoint host or a substitute for binary inspection.

## Publication/link requirements before any host is used

1. Keep the submitted supplement itself within TMLR's 100-MB limit and in its permitted PDF/ZIP form. Include a neutral short README/checksum manifest in that ZIP, not several gigabytes of weights.
2. Only add a link after the real record exists and its logged-out access has been tested. The link must not contain or redirect to a personal, project-owner, named release, institutional, or identity-bearing URL. No DOI, account link, author profile, or identifiable README should be attached during review.
3. Host paths are proposed only: flat `weights/<exact-manifest-filename>` names, with an asset-only manifest and checksums. Do not publish the source release URL or original repository metadata in the reviewer-facing record. Preserve the original per-file names/hashes for unchanged files; mark any sanitized derivative separately with its own size/hash and source-hash provenance.
4. After download, reviewers can use the already documented local-only verification/materialization interface, e.g. `python supplement.py materialize "$ARTIFACT_DIR" --role final_clean_v6_checkpoint --verify-only`, followed by the same command with `--mode hardlink` if desired. This command verifies local files and does not retrieve from a host. There is no valid network retrieval command until a real, reviewed link exists.
5. List missing caches/data separately: a weight link does not provide the TRAIN/official/LongEmbed/K-ary inputs, teacher checkpoint, or full vector caches. Keep the historical-source-drift caveat in `reproducibility/HISTORICAL_SOURCE_DRIFT.json`; do not rebind its old source digests to current files to make a restored-weight run appear to pass.

## Primary-source policy and service references

1. [TMLR Author Guidelines](https://jmlr.org/tmlr/author-guide.html) — double-blind requirement; supplementary material is limited to 100 MB, must be PDF or ZIP, and must be anonymized.
2. [TMLR Editorial Policies](https://jmlr.org/tmlr/editorial-policies.html) — double-blind process and expectations not to seek authors' identities.
3. [TMLR FAQ](https://jmlr.org/tmlr/faq.html) — reviewers should not actively search for an otherwise available preprint's authors.
4. [OpenReview upload-size FAQ](https://docs.openreview.net/getting-started/frequently-asked-questions/what-is-the-max-file-size-for-uploads-1) and [supplementary-field setup example](https://docs.openreview.net/getting-started/hosting-a-venue-on-openreview/customizing-your-submission-form/enabling-supplementary-material-upload) — platform-level upload cap and an example venue-configurable file field. The sample's 50-MB setting is not a TMLR-specific limit; TMLR's own guide says 100 MB.
5. [Anonymous GitHub FAQ](https://anonymous.4open.science/faq) — redaction scope and binary/8-MB constraints.
6. [Zenodo size-limit FAQ](https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo), [file management](https://help.zenodo.org/docs/deposit/manage-files/), [draft/restricted-link sharing](https://help.zenodo.org/docs/share/link-sharing/), [draft upload/visibility](https://help.zenodo.org/docs/deposit/create-new-upload/), and [creator requirements](https://help.zenodo.org/docs/deposit/describe-records/creators/).
7. [Hugging Face Hub storage limits](https://huggingface.co/docs/hub/en/storage-limits) and [download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

TMLR's published guide specifies the direct supplementary-file format/limit and anonymity requirements, but the sources reviewed here do not give a blanket approval or a dedicated workflow for hosting multi-gigabyte weights at an external service. An external link is therefore a conditional implementation choice, not a policy exception or a guarantee of acceptance.