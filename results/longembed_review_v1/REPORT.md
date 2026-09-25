# LongEmbed frozen-model retrieval

| Measurement | Mean | Paired 95% CI |
| --- | ---: | --- |
| 2wikimqa: first | 0.204386059 | [0.166265003, 0.245482244] |
| 2wikimqa: weighted | 0.264334253 | [0.220134257, 0.306886264] |
| 2wikimqa: unweighted | 0.275229143 | [0.229902395, 0.318888895] |
| 2wikimqa: context mean | 0.277642895 | [0.233613752, 0.322530888] |
| 2wikimqa: context cls | 0.870874581 | [0.837256379, 0.904650004] |
| 2wikimqa: learned mean | 0.276832003 | [0.231434284, 0.320731474] |
| 2wikimqa: learned minus weighted | 0.01249775 | [0.00253226895, 0.0226764774] |
| qmsum: first | 0.199553028 | [0.157787932, 0.241997579] |
| qmsum: weighted | 0.25896941 | [0.206603784, 0.314450743] |
| qmsum: unweighted | 0.253340472 | [0.201180675, 0.307969992] |
| qmsum: context mean | 0.252429274 | [0.203064036, 0.303715735] |
| qmsum: context cls | 0.406897009 | [0.362023472, 0.452278424] |
| qmsum: learned mean | 0.239020347 | [0.187419367, 0.293026122] |
| qmsum: learned minus weighted | -0.0199490635 | [-0.0298176655, -0.0100723066] |
| summ screen fd: first | 0.554134254 | [0.514497723, 0.594550028] |
| summ screen fd: weighted | 0.799833637 | [0.765912, 0.832274601] |
| summ screen fd: unweighted | 0.793316409 | [0.759435946, 0.826652311] |
| summ screen fd: context mean | 0.694223179 | [0.653971026, 0.73250312] |
| summ screen fd: context cls | 0.931410816 | [0.911289072, 0.948907754] |
| summ screen fd: learned mean | 0.772829091 | [0.736962379, 0.807754387] |
| summ screen fd: learned minus weighted | -0.0270045462 | [-0.0389299786, -0.0147902849] |
| narrativeqa: first | 0.0448702333 | [0.0370865564, 0.0537517415] |
| narrativeqa: weighted | 0.0834965455 | [0.0753747674, 0.0919711995] |
| narrativeqa: unweighted | 0.0829705999 | [0.0749395673, 0.0913384554] |
| narrativeqa: context mean | 0.062033678 | [0.0542152939, 0.070048543] |
| narrativeqa: context cls | 0.301492244 | [0.272853486, 0.329866433] |
| narrativeqa: learned mean | 0.0790156956 | [0.0706656619, 0.0877878475] |
| narrativeqa: learned minus weighted | -0.00448084987 | [-0.00854013293, -9.72914301e-05] |

All four real tasks and all candidates/queries at the pinned revision. nDCG@10 on a 0--1 scale. Mean-pool queries match all mean-based representations; CLS queries match the documented first-token context reference only. Teacher context references truncate, whereas recursive and flat full-document representations use every token. Intervals cluster queries by connected positive-document components while preserving query weighting and fixed candidate corpora. No task fitting or model selection.
