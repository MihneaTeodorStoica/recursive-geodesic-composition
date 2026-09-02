# V1 publication-system audit for V3 CLS

Base commit: `c6b3f28830e7965533e1d036c2a749836b7848b6` (`origin/main`, initial paper release).

## Final architecture available in the publication repository
- Merger: `GeodesicMerge` in `src/recursive_geodesic_composition/architecture.py`.
- Final arity: `K=4` (`arity: 4`) with contiguous balanced reduction and singleton carry handled by `contiguous_reduction_groups`.
- Embedding dimensionality follows the pinned teacher hidden size (`Alibaba-NLP/gte-modernbert-base`, 768). Final V1 configs use hidden size 1024, 8 encoder Transformer layers, 16 attention heads, dropout 0.05, `max_geodesic_angle: 0.5`, `geodesic_parameterization: tangent_expmap`, semantic token salience, and strict parent-only decoder.
- Composition: children are L2-normalized; token masses are normalized from propagated token counts; optional semantic salience adjusts mass logits; the mass-weighted anchor is moved by a learned tangent-vector exponential-map correction and renormalized.
- Recursive reduction: exact 1024-token leaves; adjacent contiguous K=4 groups are merged; trailing singleton is carried unchanged; token counts are summed and propagated.
- Decoder: `_StrictDecoder` is parent-only. It receives only the parent vector, uses learned slot positions and a Transformer encoder, predicts normalized child vectors and child-importance weights. Labels/children do not enter the decoder forward pass.
- Objectives present in `train.py`: paired teacher cosine, distribution mean and variance matching, semantic-importance regularizers (prior/delta/sharpness/slot), strict decoder reconstruction, importance, and cycle losses, recursive loss collection where targets are available, AdamW, cosine LR schedule/warmup/min LR, optional mixed precision (final V1 set to none), storage and collapse guards, bootstrap attack evaluation utilities.
- Data sources pinned in final V1 configs: FineWeb-Edu sample-10BT `87f09149...`, FineWeb sample-10BT `9bb295...`, OpenWebText `79d93...`, Wikipedia 20231101.en `b04c8...`; final V1 train source range `[140000,152000)` per corpus, development `[152000,161000)`, buffer `[161000,162000)`, official `[162000,192000)`.

## Teacher-representation defect to correct in V3
The publication repository's `FrozenTeacher.embed` used masked mean pooling over the last hidden state. The GTE ModernBERT model-card embedding path is normalized CLS: `outputs.last_hidden_state[:, 0]` followed by L2 normalization. V3 therefore treats V1 results as historical artifacts and starts a new experiment in the documented CLS space.

## Reproducible versus missing V1 training lineage
Reproducible from the publication repository: final architecture source, final configs, final evaluation scripts, official/development result files, checksums, release artifact manifest, k-ary scaling/retention validation code, and the final checkpoint-loading/evaluation semantics.

Not fully reproducible from the publication repository: the complete pre-final continuation-training lineage that produced the autoencoder/encoder initialization checkpoint referenced by final V1 configs (`initialize_encoder_from_autoencoder_checkpoint`). The final configs identify a checkpoint hash but the repository does not encode all earlier training stages, intermediate data accesses, optimizer states, or decision history needed to reconstruct that learned initialization from random seeds. V3 therefore must not claim to know or reuse that lineage and is explicitly fresh-from-random.
