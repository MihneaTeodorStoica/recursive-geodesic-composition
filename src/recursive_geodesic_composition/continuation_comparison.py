"""Paired continuation comparisons, including document-weighted AUC resampling."""
from __future__ import annotations
import numpy as np
from .safe_baselines import shared_bootstrap_means, summarize_scores


def paired_summary(scores, mask, reps, seed):
    values = {key: np.asarray(value)[mask] for key, value in scores.items()}
    draws = shared_bootstrap_means(np.column_stack(list(values.values())), reps, seed)
    return {key: summarize_scores(value, draws[:, i], "delta" in key or "gain" in key or "minus" in key)
            for i, (key, value) in enumerate(values.items())}


class DocumentWeightedAUC:
    """Exact weighted Mann--Whitney AUC with half credit for score ties.

    Each bootstrap count weights BOTH the teacher and generated observation for
    a document. Sorting is fixed once; classifiers are neither refit nor selected
    in a resample. Orientation is folded around 0.5, as in the original protocol.
    """
    def __init__(self, bundle):
        self.y = np.asarray(bundle["y"], dtype=np.int64)
        self.scores = np.asarray(bundle["scores"], dtype=np.float64)
        self.groups = np.asarray(bundle["groups"])
        if self.y.ndim != 1 or self.y.shape != self.scores.shape or self.y.shape != self.groups.shape:
            raise ValueError("invalid prediction bundle shapes")
        if not np.isfinite(self.scores).all() or set(np.unique(self.y)) != {0, 1}:
            raise ValueError("need finite predictions from both classes")
        self.documents, inverse = np.unique(self.groups, return_inverse=True)
        for group in range(len(self.documents)):
            if sorted(self.y[inverse == group].tolist()) != [0, 1]:
                raise ValueError("each document must pair exactly one teacher and one generated example")
        order = np.argsort(self.scores, kind="stable")
        self.inverse = inverse[order]
        self.positive = self.y[order] == 1
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.scores[order]) != 0) + 1]

    def raw(self, document_counts):
        counts = np.asarray(document_counts, dtype=np.float64)
        if counts.shape != self.documents.shape or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
            raise ValueError("invalid document weights")
        weight = counts[self.inverse]
        pos = np.add.reduceat(weight * self.positive, self.starts)
        neg = np.add.reduceat(weight * ~self.positive, self.starts)
        below = np.cumsum(neg) - neg
        return float(np.sum(pos * (below + .5 * neg)) / (pos.sum() * neg.sum()))

    def safe(self, document_counts):
        value = self.raw(document_counts)
        return max(value, 1 - value)


def paired_auc(source_bundles, final_bundles, reps, seed):
    if len(source_bundles) != len(final_bundles) or not source_bundles:
        raise ValueError("need checkpoint-matched source/final bundles")
    evaluators = [DocumentWeightedAUC(b) for b in [*source_bundles, *final_bundles]]
    reference = evaluators[0]
    for item in evaluators[1:]:
        if not np.array_equal(item.documents, reference.documents):
            raise ValueError("classifier holdout documents differ")
    n, k = len(reference.documents), len(source_bundles)
    points = np.array([item.safe(np.ones(n)) for item in evaluators])
    rng = np.random.default_rng(seed)
    samples = np.empty((reps, 2 * k))
    for i in range(reps):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        samples[i] = [item.safe(counts) for item in evaluators]
    def summary(point, values):
        return {"mean": float(point), "ci95": np.percentile(values, [2.5, 97.5]).tolist()}
    per_checkpoint = [{"source": summary(points[i], samples[:, i]),
                       "final": summary(points[k + i], samples[:, k + i]),
                       "final_minus_source": summary(points[k + i] - points[i], samples[:, k + i] - samples[:, i])} for i in range(k)]
    src, fin = samples[:, :k].mean(1), samples[:, k:].mean(1)
    return {"per_checkpoint": per_checkpoint,
            "three_seed_mean": {"source": summary(points[:k].mean(), src),
                                "final": summary(points[k:].mean(), fin),
                                "final_minus_source": summary(points[k:].mean() - points[:k].mean(), fin - src)},
            "holdout_documents": n, "reps": reps, "bootstrap_seed": seed,
            "fixed_observed_families_no_refitting_or_reselection": True}
