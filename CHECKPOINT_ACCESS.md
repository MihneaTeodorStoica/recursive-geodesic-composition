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

No pickle payload was executed. These are structural/raw-byte preservation proofs, **not** a claim that a Torch tensor comparison or inference ran on this packaging host. The original and derivative SHA-256 values are explicitly different in the hosted manifest. Original experiment manifests, frozen result hashes and original checkpoint identities remain untouched.

**Original-byte provenance guards will reject a sanitized warm derivative.** Do not bypass those guards or replace recorded experiment hashes. These warm intermediates are optional for inspecting the intervention state; the existing gradient-control training path regenerates them from exact source checkpoints, and the six final attached/detached models can be loaded without them.

## Remaining reproduction and anonymity boundaries

- The final clean-v6 models use **300 decoder-only + 100 joint-update continuation from exact source checkpoints**, not random initialization. Complete historical from-scratch training lineage of those source models remains unestablished.
- Teacher/corpus/task data, token/embedding/vector caches and compatible runtime/GPU resources remain external. Weight publication does not make the frozen downstream chain turnkey; historical source-fingerprint guards and create-only outputs remain. Preserve the documented 15 presentation differences, three historical recovery versions and one unresolved preparer fingerprint.
- Nine older artifacts outside the released 66-file inventory remain unavailable: three matched-pooling probes, three scaling models and three scaling retention probes. Their exact paths/hashes are retained in `CHECKPOINT_HOSTING_PLAN.md`, original study manifests and the companion hosted manifest. No byte sizes or replacement identities are invented.
- Full-document roots still have no direct full-root teacher target/AUC; classifier/bootstrap replay from saved scalars is not fresh model evaluation. Tables 4–17 and all reported numbers are unchanged.
- Author-only acquisition/anonymity audit reports and raw identifying metadata are not published. Public titles, scientific results and exact artifact hashes remain searchable; perfect anonymity against deliberate comparison or hosting operators is not claimed.
