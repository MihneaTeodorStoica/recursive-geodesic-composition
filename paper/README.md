# Paper source

**One Vector from Many: Hierarchical Long-Document Representation through Recursive Geodesic Transformer Composition**  
Mihnea-Teodor Stoica · October 2026

- [`main.tex`](main.tex): manuscript, including inline TikZ/PGFPlots figures.
- [`references.bib`](references.bib): BibTeX bibliography.
- [`figures/`](figures/): all three PNG assets bundled with the supplied archive. The current manuscript does not reference these external images.

## Build

Install a recent TeX distribution with pdfLaTeX, BibTeX, `latexmk`, and the packages used in `main.tex` (including TikZ/PGFPlots). The official `tmlr.sty` and `tmlr.bst` files are vendored unchanged from [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file). From the repository root:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This produces `paper/main.pdf` in anonymous TMLR review format. The manuscript, not the build-product PDF, is tracked. After switching from the former numeric bibliography, run `latexmk -C` before the first build to discard stale auxiliary files. Alternatively, from `paper/`, use `tectonic -X compile main.tex --hide main.aux --hide main.bbl --hide main.out --keep-logs`; the hide options ignore stale build inputs without deleting them.

For the named, arXiv-compatible preprint with the repository link and artifact citation, run:

```bash
pdflatex -jobname=preprint '\def\papreprint{1}\input{main.tex}'
bibtex preprint
pdflatex -jobname=preprint '\def\papreprint{1}\input{main.tex}'
pdflatex -jobname=preprint '\def\papreprint{1}\input{main.tex}'
```

The preprint switch uses the standard `article` class, US Letter paper with one-inch margins, single-column text, and `natbib`/`plainnat` references; it does not load the TMLR style or display a TMLR submission banner. Author identity is included both visibly and in PDF metadata. [arXiv's source guidelines](https://info.arxiv.org/help/submit_tex.html) do not prescribe a universal visual template: include the bibliography and all referenced figures/table inputs, omit build intermediates, and inspect arXiv's generated PDF before submitting. These builds have been checked locally, not submitted to arXiv.

For Tectonic, create a temporary entrypoint in `paper/` and compile it separately from `main.tex`:

```bash
printf '\\def\\papreprint{1}\\input{main.tex}\n' > One_Vector_from_Many_arxiv.tex
SOURCE_DATE_EPOCH=1792562848 tectonic -X compile One_Vector_from_Many_arxiv.tex \
  --hide One_Vector_from_Many_arxiv.aux \
  --hide One_Vector_from_Many_arxiv.bbl \
  --hide One_Vector_from_Many_arxiv.out --keep-logs
```

The v3.2 release builds used Tectonic 0.17.0 and `SOURCE_DATE_EPOCH=1792562848` (the manuscript merge timestamp), producing a 38-page named preprint and a 40-page anonymous TMLR submission. Both contain all six remedies, resolved references, and no overfull boxes; the TMLR build retains underfull-page layout warnings. Opening pages were visually inspected, and extracted submission text and PDF author metadata passed anonymity checks. The bibliography rerun-limit warning observed with Tectonic 0.15.0 did not occur in the release builds.


### v3.3-paper release

[`v3.3-paper`](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v3.3-paper) corrects the release documentation and download links. It carries all 75 assets from v3.2 unchanged, including both current PDFs and all four checksum files. The manuscript, numerical results, and trained weights are unchanged; earlier releases remain available.

The v3.2 build commands, compiler, and fixed epoch below also apply to this tag because the manuscript source is unchanged. The PDF build verification remains the v3.2 verification, not a new build or scientific evaluation. This tag additionally pins the corrected documentation.

### v3.2-paper release

[`v3.2-paper`](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v3.2-paper) pins the October 21 manuscript revision at `68c937b5907764a36301704c07970397543fcf6d`. It updates citations, the introduction roadmap, terminology, Figure 1 descriptions, and prose; the named preprint adds the contact email and October 2026 date. Experimental values and trained weights are unchanged.

The release contains 75 assets: the two current PDFs, `PAPER_FORMAT_ASSETS.sha256`, all 70 unchanged v1/v2 attachments, and the original v3 PDF/checksum. The current checksum covers only the two current PDFs. Use `ARTIFACTS.sha256`, `REVIEW_RELEASE_ASSETS.sha256`, and `PAPER_RELEASE_ASSETS.sha256` for the original v1, v2, and v3 assets respectively. Earlier releases remain intact.

To rebuild this manuscript revision, check out `v3.2-paper` and use the commands above. For the anonymous output, run from `paper/`:

```bash
SOURCE_DATE_EPOCH=1792562848 tectonic -X compile main.tex \
  --hide main.aux --hide main.bbl --hide main.out --keep-logs
```

This produces `main.pdf`; the separate preprint entrypoint produces `One_Vector_from_Many_arxiv.pdf`. The fixed epoch and compiler record provenance, not a cross-toolchain byte-identical guarantee. The source archive at `v3.2-paper` predates this documentation correction; these instructions describe its actual release build. The historical publication-check caveat below still applies; the editorial release did not rerun numerical evaluations.

### v3.1-paper release

[`v3.1-paper`](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v3.1-paper) pins the separate arXiv-compatible preprint and anonymous TMLR submission formats, together with these build instructions. Its assets include `One_Vector_from_Many_arxiv.pdf` (38 pages), `One_Vector_from_Many_tmlr_submission.pdf` (40 pages), and `PAPER_FORMAT_ASSETS.sha256`. Build from this tag using the commands above; `main.pdf` is the TMLR submission output. The release also consolidates all 70 unchanged v1/v2 attachments and the original v3 PDF/checksum. Original checksum manifests retain their original scopes; no newly trained weights or experimental results are introduced.

### v3-paper release

[`v3-paper`](https://github.com/MihneaTeodorStoica/recursive-geodesic-composition/releases/tag/v3-paper) includes the original 40-page named TMLR preprint `One_Vector_from_Many_v3.pdf` and `PAPER_RELEASE_ASSETS.sha256`, plus all 70 unchanged attachments from `v1-paper` and `v2-paper-review`. Older releases remain available; their original manifests and checksum files still apply to the legacy assets. No new experimental results or newly trained weights are introduced.

The separate arXiv-compatible and anonymous TMLR formats are released at `v3.1-paper`, not built from the older `v3-paper` tag. Older tags and their automatic source archives remain unchanged.

The following historical recipe reproduces the original named TMLR preprint from the release tag, using Tectonic 0.17.0 and `SOURCE_DATE_EPOCH=1792501521` fixed to the manuscript merge timestamp:

```bash
git switch --detach v3-paper
cd paper
printf '\\def\\papreprint{1}\\input{main.tex}\n' > One_Vector_from_Many_v3.tex
SOURCE_DATE_EPOCH=1792501521 tectonic -X compile One_Vector_from_Many_v3.tex \
  --hide One_Vector_from_Many_v3.aux \
  --hide One_Vector_from_Many_v3.bbl \
  --hide One_Vector_from_Many_v3.out --keep-logs
```

The one-line entrypoint is temporary build scaffolding, not a second manuscript. It and the generated PDF/intermediates should not be committed. The fixed epoch and compiler version record provenance, not a claim of byte-identical output on every toolchain.

Release verification: the named PDF builds without unresolved references or overfull boxes; underfull-box layout warnings remain. The opening, bibliography/artifact links, and limitation/remedy layout were inspected, and all six proposed remedies are present.

The existing publication checks (`.venv/bin/python -m pytest tests/test_*publication*.py`, from the repository root) returned **35 passed, 4 failed**. The four failures are historical `ARTIFACT_HASHES.json` assertions covering five table files: `continuation_comparison_table.tex`, `reconstruction_calibration_table.tex`, `full_population_table.tex`, `recursive_reuse_table.tex`, and `safe_baselines_table.tex`. All five numerical tabular bodies match the v2-era files; the later TMLR conversion moved captions above the tables and revised some caption wording. An expanded check matched the other 101 available artifact hashes in those four namespaces; regenerable absent `vectors.pt` files were not checked. Historical manifests remain unchanged rather than being silently re-pinned. This is not an all-green test-suite claim or a new numerical rerun.


### TMLR submission checks

Checked against the [author guide](https://jmlr.org/tmlr/author-guide.html) and official template on 2026-10-18:

- Use the unmodified official style and author-year bibliography; do not override its margins or paragraph spacing.
- Default PDF hides the author, identifying repository URL, artifact self-citation, and the two project repository commit IDs; PDF author metadata is empty. The preprint switch restores the named title, artifact links, and implementation/experiment commit IDs. Third-party model/dataset revisions and checkpoint/result SHA-256 digests remain available in both variants.
- Abstract is one paragraph; references precede appendices; all table captions, including Table 14 and included result tables, appear above their contents. Table generators retain that placement.
- There is no fixed page limit, but length must be justified; appendices are discretionary reading.
- Before submission, authors must separately verify the editorial policies, acceptance criteria, ethics guidelines, and code of conduct; maintain complete OpenReview profiles; supply the requested editor/conflict, funding, competing-interest, and human-subject information; and review the CC BY 4.0 submission license.
- Any supplementary upload must itself be anonymized and supplied as PDF/ZIP (at most 100 MB). The public repository is not an anonymous supplement; do not upload its identifying metadata unchanged. No submission, license acceptance, or supplementary upload is performed by the build.

## Import provenance

All five source files were extracted byte-for-byte from
`One_Vector_from_Many__Hierarchical_Long_Document_Representation_through_Recursive_Geodesic_Transformer_Composition.zip`
(SHA-256: `e1b8a8ed40478e02f45e530e62d0a31d9d7ed847123e8cb3d54ebed6026dd042`).
The archive was removed after extraction and byte verification.

The bundled PNGs differ from the existing frozen publication assets in [`../figures/`](../figures/); both sets are preserved separately. Importing the manuscript does not alter experimental results, model code, or citation metadata. File checksums are recorded in [`../reproducibility/hashes.txt`](../reproducibility/hashes.txt).

That statement describes the *initial import only*. Subsequent manuscript builds were validated with a locally installed Tectonic binary. See the additive [review evidence map](REVIEW_EVIDENCE_MAP.md), which distinguishes previously evaluated prefixes and separately pinned downstream data.

The `v2-paper-review` release predates the writing-coach revision and TMLR formatting cutover. Its 35-page PDF and fixed-epoch hashes describe that historical release, not a rebuild of current source. Current edits preserve reported experimental values while revising prose, hierarchy, and table presentation.

The 2026-10-20 revisions adapt the October 18 editorial comments to the current manuscript, followed by a first-reader pass: the abstract introduces the problem and method before reporting fidelity and cost; the main text explains terminology and comparison purposes before using experiment labels. Technical details remain in labeled appendices, and the combined Limitations and Future Work subsection pairs each summary-first limitation with a visibly labeled proposed remedy. The bibliography uses the official TMLR style, literal DOI identifiers, and HTTPS URLs for every entry; DOI-only entries link through `https://doi.org/`. The annotated older Word file is feedback, not the source of the revised manuscript. Publication tests check the numerical tables and underlying results without requiring duplicate values or particular wording in the manuscript prose.

The compositional-space framing makes recursive reuse the central goal: parents retain the encoder-output dimension and unit-sphere domain and can enter subsequent merges without conversion. Semantic alignment with directly encoded spans is a training objective assessed empirically, not guaranteed by this reusable interface. Pooling remains the comparator, not the motivation. The independent-review pass clarifies that headline fidelity is a single-merge result, distinguishes normalized recursive pooling from exact flat pooling, and states the inherited-checkpoint/continuation-training scope before the training recipe.

The 2026-10-21 revision applies the October 20 tracked editorial comments to the current source: parenthetical citations (with narrative citations retained where authors are sentence subjects), a complete introduction roadmap, consistent merger terminology and source-index ranges, scientific notation for very small results, serial commas, and shorter repeated descriptions. The named preprint adds the author’s contact email and uses October 2026 as the intended submission month; update this date if submission moves to another month. The anonymous TMLR output keeps the contact hidden. Figure 1 uses thick/thin arrow descriptions, limitations retain visible italic *Proposed remedy* labels, and the classifier two-sample citation uses the ICLR 2017 version. Experimental results and historical artifact manifests are unchanged.

Verification for this revision: Tectonic 0.15.0 built the 38-page named preprint and 40-page anonymous TMLR submission. Rendered title pages, Figure 1, and the limitation/remedy layout were inspected. Both logs contain no undefined references or overfull boxes; all six remedies are present in both PDFs. Extracted submission text and author metadata omit the contact identity. TMLR underfull-page warnings remain, and this compiler reports an internal bibliography-change/rerun-limit warning despite resolved citations in both outputs.
