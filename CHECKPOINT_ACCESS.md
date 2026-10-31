# Public checkpoint access

Model repository: [Hugging Face checkpoint repository](https://huggingface.co/tmlr-embedding-artifacts/recursive-geodesic-checkpoints)

**Immutable Hugging Face revision:** `9187157fa4fc6f790be1775475952f14ae33d02e`

[Anonymous source and reproduction guide](https://anonymous.4open.science/r/recursive-geodesic-composition-0F32/).

All **66 released weight artifacts**, totaling **8,110,072,711 bytes**, are available at this revision without authentication or gated-access approval: **63 exact originals plus three explicitly identified metadata-only warm-state derivatives**. The 60 files external to the small supplement total 8,062,738,309 bytes. No model tensors are placed in the supplementary ZIP beyond its six original historical probes.

## Download only the required groups

Run from the review checkout or extracted `TMLR_Supplement/` root with Python 3.12 or later; no project installation, Hub package or token is needed:

```bash
python scripts/download_checkpoints.py --group final --verify
python scripts/download_checkpoints.py --group source --group historical_probes --verify
python scripts/download_checkpoints.py --group gradient_path --verify
python scripts/download_checkpoints.py --group decoder_probes --verify
python scripts/download_checkpoints.py --group intermediate --verify
```

Each command selects only its requested groups. Use repeated `--group` to combine selections, or `--group all` explicitly for the complete inventory. `--root /path/to/writable-review-copy` installs into another evaluation checkout while reading the pinned manifest from this checkout.

| Group | Files | Hosted bytes | Scientific role |
| --- | ---: | ---: | --- |
| `final` | 3 | 1,483,558,373 | Three final clean-v6 model checkpoints. |
| `source` | 3 | 1,483,552,229 | Seed-matched continuation initialization; also needed for source/final and official-preaccess comparisons. |
| `historical_probes` | 6 | 47,334,402 | Original fitted candidate/source parent-only probes; bundled files are reused without network downloads. |
| `gradient_path` | 6 | 2,967,128,382 | Separately trained attached/detached intervention endpoints. |
| `decoder_probes` | 45 | 354,978,585 | Additional direct/bounded/unbounded TRAIN-fitted probe controls. |
| `intermediate` | 3 | 1,773,520,740 | Metadata-only sanitized shared warm states; see the provenance limitation below. |

`review_controls` is an overlapping convenience group containing the 54 gradient-path, decoder-probe and intermediate artifacts. Do not add its count to the disjoint rows above.

The helper always verifies full size/SHA-256, refuses different existing files, reuses verified local destinations/bundled probes, and uses hard links for protocol/lookup aliases where possible (copies across filesystems). `--verify` additionally rechecks every installed path. A second identical invocation does not redownload matching files. Failed downloads are never installed as checkpoints. The script sends no authentication headers and does not read local Hugging Face credentials.

## Independent manifest and manual verification

`reproducibility/HOSTED_CHECKPOINTS.json` contains the repository ID, immutable revision, hosted paths, original roles, model/probe seeds and evidence, original and hosted hashes, exact sizes, installation paths, groups and explicit transformation mappings. `MANIFEST.json` on the Hub describes the identical checkpoint payload; its SHA-256 is pinned in the companion manifest. The latter adds the enclosing revision without creating an impossible self-referential commit hash.

To download the complete repository manually with an installed Hugging Face CLI:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download \
  tmlr-embedding-artifacts/recursive-geodesic-checkpoints \
  --revision 9187157fa4fc6f790be1775475952f14ae33d02e \
  --local-dir artifacts/huggingface-checkpoints
(cd artifacts/huggingface-checkpoints && sha256sum --check SHA256SUMS)
```

The group helper is preferred for evaluation because it also places files at the frozen protocol paths. Final/source models and historical probes additionally receive `artifacts/v1-paper/<asset_filename>` lookup paths. Review-control artifacts receive their recorded `results/...` protocol paths. The original flat-directory materializers remain exact-original-byte workflows, not sanitized-warm download interfaces.

Every one of the **66 checkpoint responses** was independently downloaded in full without authentication and matched its published size and SHA-256: **8,110,072,711 bytes verified**. Actual PyTorch ZIP payloads—not LFS/Xet pointers—were checked. Remote filenames/sizes were reconciled with the manifest; card, manifests, checksum list and public commit authorship were inspected separately. A real cold helper invocation downloaded all three final models and verified six protocol/lookup paths; the repeated invocation downloaded zero files. Checksums establish byte agreement, not independent authenticity or scientific correctness.

## Warm-state transformation: scientific state preserved, serialized identity changed

The original three warm checkpoints contain identifying location metadata and are **not** anonymously redistributed unchanged. Each hosted derivative changes only the personal location prefix in `initialization.initialization_checkpoint`, retaining the original source filename and relative artifact suffix. String length, ZIP member offsets and file size are preserved; the corresponding CRC fields are updated.

Static comparisons established, per warm checkpoint:

- All tensor-storage bytes are unchanged, including model and optimizer tensors.
- All tensor rebuild descriptors, shapes, dtypes, storage offsets and strides are unchanged: 132 model descriptors and 57 optimizer descriptors.
- Every numeric/non-string pickle argument and every non-identifying string is unchanged; no optimizer/scheduler/RNG/scientific state is dropped or reinitialized.
- All archive bytes outside the targeted string payload and required CRC fields are unchanged; the rewritten archives pass CRC and metadata scans.

Initial packaging checks were static. Subsequent isolated CPU verification loaded all 66 hosted files with `weights_only=True`, dynamically compared all 191 tensors in each original/derivative warm pair, strictly loaded 12 final/source/gradient models, and exercised merge, recursion and decoder paths. All three warm pairs produced bitwise-identical synthetic outputs. This proves loadability/state preservation, not reported metric replication or GPU equivalence. Original and derivative SHA-256 values remain explicitly different; frozen experiment identities are untouched.

**Original-byte provenance guards will reject a sanitized warm derivative.** Do not bypass those guards or replace recorded experiment hashes. These warm intermediates are optional for inspecting the intervention state; the existing gradient-control training path regenerates them from exact source checkpoints, and the six final attached/detached models can be loaded without them.

## Recovered numeric caches: separate immutable release

**Cache revision:** `9e4c8e02375f66b3cf9918a20eb81446f996620e`. The 66-weight revision above is unchanged.

`reproducibility/HOSTED_CACHES.json` pins the separate Hub `CACHES_MANIFEST.json`, six bundles and all 8,075 extracted-file hashes. Bundles total **911,050,856 bytes**, expanding to **1,089,336,987 bytes**. All **8,032 PT files**, per-corpus indexes and sidecars are exact recovered bytes.

| Extracted directory | Contents |
| --- | --- |
| `cache_kary4_strict_autoencoder_v6_clean_train` | Original clean TRAIN leaves and teacher targets; manifest matches the historical `LOCAL_LARGE_ARTIFACTS.sha256` pin. |
| `cache_kary4_strict_autoencoder_v6_clean_development` | Development safe/full leaves, full indexes and shards. |
| `cache_kary4_strict_autoencoder_official_test_v6` | Official safe/full leaves; two explicit anonymous metadata derivatives. |
| `cache_kary4_1024_expanded_source_seed424_v1` | Earlier expanded-source TRAIN/validation leaves and targets. |
| `cache_kary4_leaf510_vs1024_v1_leaf1024` | Earlier leaf1024 TRAIN/validation tensors/sidecars; no original root manifest/COMPLETE was recovered. |
| `cache_kary4_1024_validation3_strict_autoencoder_seed424_v2` | Separate validation3 leaves and targets. |

Download from a **fresh writable review copy** with the installed Hugging Face CLI. These commands use no account/token, check compressed hashes before extraction, and refuse to overwrite existing cache files:

```bash
REVIEW_ROOT="$PWD"
CACHE_DOWNLOAD="$REVIEW_ROOT/artifacts/recovered-cache-download"
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download \
  tmlr-embedding-artifacts/recursive-geodesic-checkpoints \
  --revision 9e4c8e02375f66b3cf9918a20eb81446f996620e \
  CACHES_MANIFEST.json CACHE_SHA256SUMS \
  caches/cache_kary4_strict_autoencoder_v6_clean_train.tar.gz \
  caches/cache_kary4_strict_autoencoder_v6_clean_development.tar.gz \
  caches/cache_kary4_strict_autoencoder_official_test_v6.tar.gz \
  caches/cache_kary4_1024_expanded_source_seed424_v1.tar.gz \
  caches/cache_kary4_leaf510_vs1024_v1_leaf1024.tar.gz \
  caches/cache_kary4_1024_validation3_strict_autoencoder_seed424_v2.tar.gz \
  --local-dir "$CACHE_DOWNLOAD" || exit 1
(cd "$CACHE_DOWNLOAD" && sha256sum --check CACHE_SHA256SUMS) || exit 1
for archive in "$CACHE_DOWNLOAD"/caches/*.tar.gz; do
  tar --keep-old-files --no-same-owner -xzf "$archive" -C "$REVIEW_ROOT" || exit 1
done
```

For a selective download, omit the unneeded bundle filenames and use `sha256sum --ignore-missing --check CACHE_SHA256SUMS`. Extract only the selected bundle. The archive root is its original logical cache-directory name. Extraction is storage recovery, not execution of a scientific protocol. Do not run create-only preparation over a restored directory.

Only official `CACHE_MANIFEST.json` and `COMPLETE` differ: the identifying implementation-parent label becomes `anonymous-source-snapshot-01`, and COMPLETE points to that derivative manifest. The original manifest digest `0c5d876b4fb9c3285c2993a8c895fc1b60040e1fa04e83bba672bf154b9b8abb` becomes `833d3cf216ac6349be0b21b070e92e0f39063d59623d499e5cd67f54d108d0ed`. Original/new hashes and sizes for both files are explicit in the inventory outside the cache root; no scientific tensor is rewritten.

**The official derivative is internally complete, not historical-guard compatible.** Frozen official results still record the original manifest digest. Original config paths/hashes and some provenance hashes differ from the normalized review layout too. Keep those historical identities and guards unchanged. The recommended [public cached replay](REPRODUCING.md#public-cached-replay-recommended-reviewer-workflow) verifies a separately pinned release contract, records both historical and normalized metadata identities, and writes new results under `replays/`; it does not make the derivative an original-byte artifact. The original TRAIN manifest uses the earlier preparation config, preceding the documented 120/120 to 80/80 probe-availability recovery; both identities remain intact.

Cache-release inspection covered all 8,075 original file hashes and 12,049 manifest/shard references; complete static metadata/CRC inspection of 8,032 pickle streams and 285,485 ZIP members; all 8,032 finite CPU `weights_only=True` loads; and one real row from each directory through the published final model's recursive reducer and strict decoder. That inspection alone is compatibility evidence, not fresh aggregate metrics. Separately executed cached-replay evidence is reported in VALIDATION.md; neither check recovers all study-specific token/vector inputs.

### Attribution, terms and linkability

The inventory records pinned FineWeb/FineWeb-Edu (`ODC-BY`), OpenWebText (`CC0-1.0`), Wikipedia (`CC-BY-SA-3.0`/`GFDL`) and GTE teacher (`Apache-2.0`) sources, upstream URLs and license links. The [cache release card](https://huggingface.co/tmlr-embedding-artifacts/recursive-geodesic-checkpoints/blob/9e4c8e02375f66b3cf9918a20eb81446f996620e/README.md) credits their maintainers and source contributors. Retain applicable source terms and attribution/share-alike obligations; project software/model terms do not replace them. Dataset-card labels are not a guarantee about every underlying web-page right.

No recognized raw text or token sequences were found in the published caches. Document IDs, source indexes and text/token hashes remain for provenance and can link to upstream documents; embeddings can retain input information. No source-document privacy or perfect author-unlinkability guarantee is made.

## Recovered source lineage: partial, not from scratch

`reproducibility/RECOVERED_SOURCE_LINEAGE.json` binds the three seeds through initial encoder (selected step 1200), continued encoder (step 1200), hosted source autoencoder (step 400), and hosted clean-v6 final (step 400). Fifteen exact primary configurations/status/freeze/candidate records are retained under `reproducibility/recovered_source_lineage/` as **non-active historical evidence**; six configuration hashes match their recorded pins.

The six earlier encoder tensors, source per-step JSONLs, historical trainer byte identities and complete RNG/execution bindings remain unestablished or absent. These records improve the lineage evidence but do not recover missing weights or prove complete from-scratch reproducibility. No scientific code, frozen config, result or checkpoint identity is changed.

## Remaining reproduction and anonymity boundaries

- The final clean-v6 models use **300 decoder-only + 100 joint-update continuation from exact source checkpoints**, not random initialization. The recovered historical records establish a partial earlier chain, not complete from-scratch training.
- Six numeric cache directories are now externally hosted as described above. Teacher/raw corpus/task inputs, other study-specific token/vector caches and compatible resources remain external. Historical source-fingerprint guards and create-only outputs remain; preserve the documented 15 presentation differences, three historical recovery versions and one unresolved preparer fingerprint.
- Nine older artifacts outside the released 66-file inventory remain unavailable: three matched-pooling probes, three scaling models and three scaling retention probes. Their exact paths/hashes are retained in `CHECKPOINT_HOSTING_PLAN.md`, original study manifests and the companion hosted manifest. No byte sizes or replacement identities are invented.
- Full-document roots still have no direct full-root teacher target/AUC; classifier/bootstrap replay from saved scalars is not fresh model evaluation. Tables 4–17 and all reported numbers are unchanged.
- Author-only acquisition/anonymity audit reports and raw identifying metadata are not published. Public titles, scientific results and exact artifact hashes remain searchable; perfect anonymity against deliberate comparison or hosting operators is not claimed.
