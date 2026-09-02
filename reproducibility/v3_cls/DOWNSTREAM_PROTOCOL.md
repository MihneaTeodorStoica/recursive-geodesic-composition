# V3 CLS downstream protocol

This post-hoc evaluation does not alter or reopen the official V3 experiment. Its complete executable specification is hashable at `scripts/v3_cls/downstream/protocol.json`.

## Representations and shared cache

The pinned teacher is `Alibaba-NLP/gte-modernbert-base` at revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. Every nonempty document is tokenized without special tokens, split into exact contiguous 1024-token leaves (retaining the final short leaf), and each leaf is embedded once as normalized CLS. Canonical empty cleaned documents are preserved as one normalized CLS/SEP-only empty-sequence leaf with zero raw-token mass; the weighted mean explicitly returns that sole leaf. One concatenated leaf tensor and one offsets vector per dataset split are materialized. The three recursive checkpoints, token-length-weighted normalized mean, and unweighted normalized mean all consume these same tensor bytes. Evaluation verifies artifact and raw tensor SHA-256 hashes before use.

The recursive reduction is deterministic contiguous K=4 with singleton carry, matching V3. Checkpoints are the final step-2400 artifacts: seed 424242 `adversarial_v2_candidate` and seeds 271828/314159 `adversarial_v2_final`, each pinned by SHA-256.

A direct teacher vector is computed only if the complete raw-token document fits the teacher (`<=8190` raw tokens plus CLS/SEP). A task includes this baseline only if every row in every required split is valid. Otherwise only exact coverage is reported; no row is silently dropped and all compared methods retain identical samples.

## LexGLUE ECtHR-A

`coastalcph/lex_glue`, configuration `ecthr_a`, revision `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c`; canonical train (9000), validation (1000), and test (1000) splits. Paragraphs are joined with two newlines. The target is genuinely multilabel: ten stored violation indicators plus a derived `no_violation` indicator for empty stored label sets (the dataset card's 10+1 convention).

For each representation independently, `OneVsRestClassifier(LogisticRegression)` is trained on train for each fixed C in `[0.01, 0.1, 1, 10, 100]`. Validation micro-F1 over all 11 labels selects C, with smaller C winning exact ties. Logistic prediction uses its standard 0.5 threshold. The selected C is refit on train+validation, then evaluated once on test. Reported standard sklearn metrics include 11-label micro/macro/samples F1, subset accuracy, per-label precision/recall/F1/support, and supplemental violation-only 10-label micro/macro F1.

## 20 Newsgroups

`SetFit/20_newsgroups` revision `f1b91292074e7cfb69be58b642d583ec262f30ed`, whose text is the headers/footers/quotes-removed sklearn form. Canonical train then test are concatenated (18,846 rows), with no filtering.

Each representation receives identical `KMeans(n_clusters=20, init="k-means++", n_init=20, max_iter=300, algorithm="lloyd", random_state=20260902)` parameters and identical ordered samples. Metrics are sklearn normalized mutual information (`average_method="arithmetic"`) and adjusted Rand index.

## Fail-closed behavior

Dataset revisions, teacher, checkpoints, cache artifacts, and protocol are pinned. Malformed/empty rows, missing labels, count differences, non-finite outputs, hash changes, and existing create-only destinations terminate the run rather than dropping or overwriting data.
