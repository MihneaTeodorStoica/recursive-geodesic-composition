"""Parent-copy and within-document slot-assignment calibration; no model fitting."""
from __future__ import annotations

import itertools
import numpy as np

from .safe_baselines import shared_bootstrap_means, summarize_scores, validate_vectors

METHODS = ("parent_copy", "strict_decoder", "frozen_probe", "weighted_copy", "unweighted_copy")
PERMUTATIONS = np.asarray(list(itertools.permutations(range(4))), dtype=np.int64)
DERANGEMENTS = np.all(PERMUTATIONS != np.arange(4), axis=1)
CONTRASTS = {
    "strict_minus_parent_copy": ("strict_decoder", "parent_copy"),
    "probe_minus_parent_copy": ("frozen_probe", "parent_copy"),
    "strict_minus_weighted_copy": ("strict_decoder", "weighted_copy"),
    "probe_minus_weighted_copy": ("frozen_probe", "weighted_copy"),
    "strict_minus_unweighted_copy": ("strict_decoder", "unweighted_copy"),
    "probe_minus_unweighted_copy": ("frozen_probe", "unweighted_copy"),
}


def cross_cosines(predictions, children):
    """[document,predicted_slot,target_slot] cosines, scored in float64."""
    predictions, children = np.asarray(predictions), np.asarray(children)
    if predictions.ndim != 3 or predictions.shape != children.shape or predictions.shape[1] != 4:
        raise ValueError("expected equal [documents,4,dimension] prediction and target arrays")
    arrays = []
    for name, value in (("predictions", predictions), ("children", children)):
        x = validate_vectors(value.reshape(-1, value.shape[-1]), name).astype(np.float64).reshape(value.shape)
        arrays.append(x / np.linalg.norm(x, axis=-1, keepdims=True))
    return np.clip(np.einsum("bid,bjd->bij", *arrays), -1, 1)


def assignment_scores(cross, zero_tolerance=1e-12):
    """Exhaust all 24 assignments, with a separate nine-derangement control.

    Predictions and their input parent are fixed. Only the target-slot labels
    change; permutations are NOT independent statistical observations.
    """
    cross = np.asarray(cross, dtype=np.float64)
    if cross.ndim != 3 or cross.shape[1:] != (4, 4) or not len(cross) or not np.isfinite(cross).all():
        raise ValueError("expected finite [documents,4,4] cross-cosine matrices")
    if np.any(np.abs(cross) > 1 + 1e-12):
        raise ValueError("cross-cosine outside [-1,1]")
    scores = cross[:, np.arange(4)[None, :], PERMUTATIONS].mean(-1)
    ordered = scores[:, 0]
    shuffled, deranged = scores.mean(1), scores[:, DERANGEMENTS].mean(1)
    def difference(a, b):
        delta = a - b
        return np.where(np.abs(delta) <= zero_tolerance, 0., delta)
    best = scores.max(1)
    return {"ordered": ordered, "shuffled_all24": shuffled, "shuffled_deranged9": deranged,
            "ordered_minus_shuffled": difference(ordered, shuffled),
            "ordered_minus_deranged": difference(ordered, deranged),
            "oracle_best_assignment": best, "oracle_best_minus_ordered": difference(best, ordered)}


def reconstruction_summary(scores, mask, reps, seed):
    columns = {f"{method}/{metric}": np.asarray(values)[mask]
               for method in METHODS for metric, values in scores[method].items()}
    differences = {name: scores[a]["ordered"][mask] - scores[b]["ordered"][mask]
                   for name, (a, b) in CONTRASTS.items()}
    combined = columns | differences
    samples = shared_bootstrap_means(np.column_stack(list(combined.values())), reps, seed)
    summaries = {key: summarize_scores(value, samples[:, i], key in differences or "minus" in key)
                 for i, (key, value) in enumerate(combined.items())}
    return {"methods": {method: {metric: summaries[f"{method}/{metric}"] for metric in scores[method]}
                        for method in METHODS},
            "contrasts": {name: summaries[name] for name in differences}}


def geometry_calibration(parent_child_cosines, residual_norm_bound=.25):
    """Loose spherical-cap oracle ceiling, not an achievable decoder score.

    Ignores the tighter coordinate box and the information available to a
    parent-only decoder. Positive outside-cap fraction rules out exact recovery.
    """
    values = np.asarray(parent_child_cosines, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("expected four finite parent-child cosines per document")
    if not 0 < residual_norm_bound < 1 or np.any(np.abs(values) > 1 + 1e-12):
        raise ValueError("invalid radius/cosines")
    radius = float(np.arcsin(residual_norm_bound))
    angles = np.arccos(np.clip(values, -1, 1))
    return {"fraction_children_outside_cap": (angles > radius).mean(1),
            "loose_cap_cosine_ceiling": np.cos(np.maximum(angles - radius, 0)).mean(1)}
