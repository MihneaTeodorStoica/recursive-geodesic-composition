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

This produces `paper/main.pdf`. Without `latexmk`, run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice from this directory. Clean auxiliary files with `latexmk -c`; use `latexmk -C` to remove the PDF as well. Build products are ignored by Git.

## Import provenance

All five source files were extracted byte-for-byte from
`One_Vector_from_Many__Hierarchical_Long_Document_Representation_through_Recursive_Geodesic_Transformer_Composition.zip`
(SHA-256: `e1b8a8ed40478e02f45e530e62d0a31d9d7ed847123e8cb3d54ebed6026dd042`).
The archive was removed after extraction and byte verification.

The bundled PNGs differ from the existing frozen publication assets in [`../figures/`](../figures/); both sets are preserved separately. Importing the manuscript does not alter experimental results, model code, or citation metadata. File checksums are recorded in [`../reproducibility/hashes.txt`](../reproducibility/hashes.txt).

The PDF build has not been verified in the import environment because neither `latexmk` nor `pdflatex` is installed.
