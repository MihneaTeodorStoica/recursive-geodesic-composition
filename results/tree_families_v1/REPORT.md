# Tree and segmentation robustness

| Measurement | Mean | Paired 95% CI |
| --- | ---: | --- |
| tree 1 delta original | -0.000573045479 | [-0.000639793947, -0.00050610162] |
| tree 2 delta original | -0.00067827924 | [-0.000751335068, -0.000605999371] |
| tree 3 delta original | -0.000461309293 | [-0.00053552957, -0.000384881333] |
| tree 4 delta original | -0.000665807425 | [-0.000737394292, -0.000589248316] |
| tree 5 delta original | -0.000605462834 | [-0.000685051586, -0.000525328834] |
| 1024 k2 delta weighted | 0.00561229619 | [0.00532517165, 0.00590471314] |
| 1024 k4 delta weighted | 0.00607360549 | [0.00579754519, 0.00634502386] |
| 512 k2 delta weighted | 0.0125844189 | [0.0120212744, 0.0131137799] |
| 512 k4 delta weighted | 0.0129120964 | [0.012371576, 0.0134347762] |
| 256 k2 delta weighted | 0.0250011975 | [0.0240352522, 0.0258926347] |
| 256 k4 delta weighted | 0.0243216861 | [0.0234052285, 0.0252059595] |

Same 606 texts and direct teacher targets. Five binary trees preserve original leaf order. Segmentation at 512/256 tokens changes leaf representations as well as reduction depth; 256 is below the configured training chunk-size range. No depth-only causal interpretation is made. All methods within a segmentation share the same new leaves; flat pools normalize only once.
