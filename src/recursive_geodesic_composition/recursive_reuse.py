"""Four-leaf tree/reuse stress test and descriptive population accounting.

The model is unchanged: a binary merge is its normal four-slot call with two
active slots. Tree depth AND active arity change; this is not a depth-only effect.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .safe_baselines import shared_bootstrap_means, summarize_scores, validate_children

CONTRASTS = {
    "recursive_minus_direct": ("recursive_binary", "direct_four"),
    "direct_minus_weighted": ("direct_four", "flat_token_weighted"),
    "recursive_minus_weighted": ("recursive_binary", "flat_token_weighted"),
    "direct_minus_unweighted": ("direct_four", "flat_unweighted"),
    "recursive_minus_unweighted": ("recursive_binary", "flat_unweighted"),
}


def validate_four(children, lengths):
    validate_children(children, lengths)
    if children.shape[1] != 4 or not torch.all(lengths > 0):
        raise ValueError("tree stress requires exactly four active original children")


def binary_reuse_four(model, children, lengths, return_intermediates=False):
    """Compute f2(f2(x1,x2),f2(x3,x4)) with exact propagated token masses.

    Inputs are prevalidated [B,4,D] / [B,4] tensors. Ordered leaf pairs are
    flattened in document-major order; both pairs are merged in a 2B batch.
    A second B-sized call merges their two generated parents. There are three
    merge nodes per document, with a two-level dependency, not a new model.
    """
    if children.ndim != 3 or children.shape[1] != 4 or lengths.shape != children.shape[:2]:
        raise ValueError("expected [B,4,D] original leaves and [B,4] counts")
    if model.arity != 4:
        raise ValueError("reuse must use the same final four-slot merger")
    batch, _, dimension = children.shape
    pairs = F.pad(children.reshape(2 * batch, 2, dimension), (0, 0, 0, 2))
    pair_lengths = F.pad(lengths.reshape(2 * batch, 2), (0, 2))
    intermediate = model.merge(pairs, pair_lengths).reshape(batch, 2, dimension)
    intermediate_lengths = lengths.reshape(batch, 2, 2).sum(2)
    root = model.merge(F.pad(intermediate, (0, 0, 0, 2)), F.pad(intermediate_lengths, (0, 2)))
    if return_intermediates:
        return root, intermediate, intermediate_lengths
    return root


def summarize_tree_scores(scores, mask, reps, seed, loss_thresholds):
    values = {key: np.asarray(value)[mask] for key, value in scores.items()}
    differences = {name: values[a] - values[b] for name, (a, b) in CONTRASTS.items()}
    combined = values | differences
    bootstrap = shared_bootstrap_means(np.column_stack(list(combined.values())), reps, seed)
    summaries = {name: summarize_scores(value, bootstrap[:, index], name in differences)
                 for index, (name, value) in enumerate(combined.items())}
    loss = -differences["recursive_minus_direct"]
    return {
        "metrics": {key: summaries[key] for key in values},
        "contrasts": {key: summaries[key] for key in differences},
        "degradation_direct_minus_recursive": summarize_scores(loss) | {
            "definition": "cos(direct,teacher) - cos(recursive,teacher); positive is degradation",
            "fraction_degraded": float((loss > 0).mean()),
            "fraction_improved": float((loss < 0).mean()),
            "fraction_tied": float((loss == 0).mean()),
            "fraction_loss_above": {str(cutoff): float((loss > cutoff).mean()) for cutoff in loss_thresholds},
        },
    }


def k4_accounting(tokens):
    """Leaf count, maximum reduction depth and actual nonsingleton merge nodes."""
    if isinstance(tokens, bool) or int(tokens) != tokens or tokens <= 0:
        raise ValueError("token count must be a positive integer")
    leaves = (int(tokens) + 1023) // 1024
    count, depth, merges = leaves, 0, 0
    while count > 1:
        full, remainder = divmod(count, 4)
        merges += full + int(remainder >= 2)
        count = full + int(remainder > 0)
        depth += 1
    return leaves, depth, merges


def population_summary(metadata, upper_bounds, context_tokens):
    if not metadata or len({row["id"] for row in metadata}) != len(metadata):
        raise ValueError("population needs nonempty unique document identities")
    if not upper_bounds or upper_bounds[-1] is not None:
        raise ValueError("final length bin must be open-ended")
    finite = upper_bounds[:-1]
    if any(not isinstance(bound, int) or bound <= 0 for bound in finite) or finite != sorted(set(finite)):
        raise ValueError("length-bin upper bounds must be positive and strictly increasing")
    tokens = np.asarray([row["full_token_count"] for row in metadata], dtype=np.int64)
    if np.any(tokens < 1025):
        raise ValueError("unexpected short document in the safe-selected full population")
    records = [k4_accounting(int(count)) for count in tokens]
    leaves, depths, merges = (np.asarray([row[index] for row in records]) for index in range(3))
    bins, lower = [], 1025
    for upper in upper_bounds:
        mask = tokens >= lower
        if upper is not None:
            mask &= tokens <= upper
        count = int(mask.sum())
        bins.append({"lower_inclusive": lower, "upper_inclusive": upper,
                     "documents": count, "fraction": count / len(tokens),
                     "depth_counts": {str(k): int(v) for k, v in sorted(Counter(depths[mask].tolist()).items())}})
        if upper is not None:
            lower = upper + 1
    if sum(row["documents"] for row in bins) != len(tokens):
        raise AssertionError("length bins did not partition the population")
    return {
        "documents": len(tokens), "tokens": int(tokens.sum()), "leaves": int(leaves.sum()),
        "internal_merges": int(merges.sum()), "mean_tokens": float(tokens.mean()),
        "mean_leaves": float(leaves.mean()), "maximum_tokens": int(tokens.max()),
        "maximum_leaves": int(leaves.max()), "maximum_depth": int(depths.max()),
        "token_histogram": {str(k): int(v) for k, v in sorted(Counter(tokens.tolist()).items())},
        "leaf_histogram": {str(k): int(v) for k, v in sorted(Counter(leaves.tolist()).items())},
        "depth_histogram": {str(k): int(v) for k, v in sorted(Counter(depths.tolist()).items())},
        "length_bins": bins,
        "context_threshold_tokens": context_tokens,
        "documents_strictly_above_context": int((tokens > context_tokens).sum()),
        "fraction_strictly_above_context": float((tokens > context_tokens).mean()),
        "documents_at_context_threshold": int((tokens == context_tokens).sum()),
        "documents_with_recursive_reuse": int((depths >= 2).sum()),
        "role": "descriptive full-document counts; no new full-root teacher comparison",
    }
