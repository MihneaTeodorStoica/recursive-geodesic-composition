# Frozen V4 downstream suite protocol

This addendum extends the completed frozen V4 experiment without retraining or changing any checkpoint. It was frozen before creating any new task embedding cache or opening new-task model performance.

LongEmbed is pinned to immutable revision `10039a580487dacecf79db69166e17ace3ede392`. The MTEB task metadata revision `6e346642246bfb4928c560ee08640dc84d074e8c` was independently inspected; every benchmark JSONL has the same content identity and size at both revisions (`LONGEMBED_REVISION_AUDIT.json`). The teacher remains normalized documented CLS from the pinned GTE checkpoint.

The four real-world tasks are NarrativeQA, QMSum, 2WikiMultihopQA (`2wikimqa` config), and SummScreenFD. Needle and Passkey are feasible and will be run at their eight unmodified context conditions. NFCorpus remains the short control.

Each dataset has one create-only leaf cache. Every method reads the same float32 normalized CLS leaves, token lengths, IDs, and document text. Leaves are contiguous 1,024 raw tokenizer-token chunks with the final short leaf retained. Queries are direct documented CLS. Recursive outputs use only the three frozen Euclidean-residual checkpoints. Baselines are token-weighted and unweighted means.

Real tasks rank the full official corpus. Synthetic tasks follow the LongEmbed/MTEB loader semantics: each `context_length` is an independent 100-document/50-query retrieval condition. Direct teacher is allowed only for a complete task/condition whose full candidate corpus and queries fit without truncation.

Metrics use `pytrec_eval` for nDCG and recall with deterministic ID tie-breaking; MRR@10 is computed from positive qrels over the same rankings. Synthetic main score is official nDCG@1, while nDCG@10, Recall@10/100, and MRR@10 are also retained. Fixed bins were declared from corpus distributions before evaluation; synthetic context values are native benchmark strata. Query bootstrap uses 1,000 replicates.

ID, qrel, cache-hash, score-shape, finite-value, or metric query-drop failures abort evaluation. No post-result bin changes, model tuning, checkpoint selection, or dataset exclusions are permitted.
