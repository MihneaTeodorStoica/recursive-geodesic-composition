# Paper source

**One Vector from Many: Hierarchical Long-Document Representation through Recursive Geodesic Transformer Composition**  
Mihnea-Teodor Stoica · September 2026

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

For a named preprint with the repository link and artifact citation, run:

```bash
pdflatex -jobname=preprint '\def\papreprint{1}\input{main.tex}'
bibtex preprint
pdflatex -jobname=preprint '\def\papreprint{1}\input{main.tex}'
pdflatex -jobname=preprint '\def\papreprint{1}\input{main.tex}'
```

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

The 2026-10-20 revisions adapt Nancy’s October 18 Word comments to the current manuscript, followed by a first-reader pass: the abstract introduces the problem and method before reporting fidelity and cost; the main text explains terminology and comparison purposes before using experiment labels. Technical details remain in labeled appendices, and an explicit Future Work subsection develops the research directions motivated by the limitations. The bibliography uses the official TMLR style, literal DOI identifiers, and HTTPS URLs for every entry; DOI-only entries link through `https://doi.org/`. The annotated older Word file is feedback, not the source of the revised manuscript. Publication tests check the numerical tables and underlying results without requiring duplicate values or particular wording in the manuscript prose.
