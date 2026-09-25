# Autonomous completion plan

Authorized after improvement 4 was pushed (`17b7fa0`). All new studies are
additive, with frozen choices before their outcomes, no replacement of official
or earlier post-review results, and all directions retained.

5. **Input-order control**: exhaust all 24 leaf/count permutations on the same
   606 exact-four prefixes, plus equal-mass and fixed-parent controls. Test
   whether predicted slots follow the input permutation, not just original
   position priors. Frozen built-in decoders and released probes; all seeds.
6. **Tree and segmentation robustness**: all five ordered binary trees on the
   original four leaves; 512/256-token resegmentations of the same text with
   binary/K4 reduction and matched flat pooling. Teacher targets stay fixed.
   Depth, grouping and segmentation remain distinct confounds, not hidden.
7. **Probe-family control**: new, matched TRAIN-only bounded, unbounded-residual
   and non-residual MLP probes for each final representation and both pools.
   Identical fixed budgets and schedules; three probe initializations; copy and
   TRAIN-only slot-centroid controls. No historical probe identity claim.
8. **Downstream utility**: all four real-world LongEmbed retrieval tasks at a
   pinned dataset revision, all documents and queries. Frozen final mergers,
   both flat pools, first chunk, context-prefix teacher and documented CLS
   interface reference. Query/document representations must use matching
   interfaces. Report all tasks, length/depth accounting and conditional paired
   cluster intervals; no tuning on relevance labels.
9. **Decoder-gradient intervention**: a new controlled continuation on the
   same new TRAIN-prefix cache, with identical warm starts and minibatches.
   Compare attached versus detached decoder-parent gradient paths. Preserve
   teacher/distribution/encoder-regularizer terms and optimization budgets.
   This isolates a gradient path in this NEW regime, not historical clean-v6
   training or all effects of using a decoder.
10. **Manuscript/reproduction cleanup**: reconcile abstract, claims, limitations,
    future work and READMEs; add a consolidated evidence/limitations map; test,
    audit, compile and inspect the complete paper.

The open-ended research directions (other languages/backbones, complete
from-scratch lineage, geometry/arity scaling at matched total training budgets)
are not silently converted into positive claims. They remain limitations unless
actually tested. Each completed item will state precisely what remains unknown.
