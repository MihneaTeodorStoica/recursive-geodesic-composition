# Paper source

**One Vector from Many: Hierarchical Long-Document Representation through Recursive Geodesic Transformer Composition**  
Mihnea-Teodor Stoica · September 2026

- [`main.tex`](main.tex): manuscript, including inline TikZ/PGFPlots figures.
- [`references.bib`](references.bib): BibTeX bibliography.
- [`figures/`](figures/): all three PNG assets bundled with the supplied archive. The current manuscript does not reference these external images.

## Build

Install a recent TeX distribution with pdfLaTeX, BibTeX, `latexmk`, and the packages used in `main.tex` (including TikZ/PGFPlots and the `plainurl` bibliography style, supplied by `urlbst`). From the repository root:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This produces `paper/main.pdf`. For the SHA-256-identical reviewed release PDF run `SOURCE_DATE_EPOCH=1790294400 tectonic -X compile main.tex` (including BibTeX; unset epochs produce different PDF metadata), or use `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Clean auxiliary files with `latexmk -c`; use `latexmk -C` to remove the PDF as well. Build products are ignored by Git. The reviewed 35-page PDF was independently built with Tectonic without warnings, checked visually on the abstract, new-results, discussion, conclusion and appendix pages, and distributed as a separate `v2-paper-review` release asset. Two builds at the fixed epoch produced the same PDF hash. The manuscript, not the build-product PDF, is the tracked paper source.

## Import provenance

All five source files were extracted byte-for-byte from
`One_Vector_from_Many__Hierarchical_Long_Document_Representation_through_Recursive_Geodesic_Transformer_Composition.zip`
(SHA-256: `e1b8a8ed40478e02f45e530e62d0a31d9d7ed847123e8cb3d54ebed6026dd042`).
The archive was removed after extraction and byte verification.

The bundled PNGs differ from the existing frozen publication assets in [`../figures/`](../figures/); both sets are preserved separately. Importing the manuscript does not alter experimental results, model code, or citation metadata. File checksums are recorded in [`../reproducibility/hashes.txt`](../reproducibility/hashes.txt).

That statement describes the *initial import only*. Subsequent manuscript builds were validated with a locally installed Tectonic binary. See the additive [review evidence map](REVIEW_EVIDENCE_MAP.md), which distinguishes previously evaluated prefixes and separately pinned downstream data.
