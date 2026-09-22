"""Pure, matched safe-arm analysis helpers; no dataset or checkpoint access."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

BASELINES = ("first_chunk", "unweighted_mean", "token_weighted_mean")
LEARNED = ("learned_reference", "learned_parent")
CONTRASTS = {
    "parent_minus_weighted": ("learned_parent", "token_weighted_mean"),
    "parent_minus_unweighted": ("learned_parent", "unweighted_mean"),
    "reference_minus_weighted": ("learned_reference", "token_weighted_mean"),
    "parent_minus_reference": ("learned_parent", "learned_reference"),
}


def validate_children(children: torch.Tensor, lengths: torch.Tensor) -> None:
    if children.ndim != 3 or lengths.shape != children.shape[:2]:
        raise ValueError("expected [B,K,D] children and [B,K] token counts")
    if not torch.isfinite(children).all() or not torch.isfinite(lengths).all():
        raise ValueError("nonfinite child or token count")
    active = lengths > 0
    if (lengths < 0).any() or (active.sum(1) < 2).any():
        raise ValueError("need at least two active children with nonnegative counts")
    if (active & (~active).cummax(dim=1).values).any():
        raise ValueError("active children must be a contiguous prefix")
    if (children.norm(dim=-1)[active] <= 1e-12).any():
        raise ValueError("zero-norm active child")


def baseline_parent(children: torch.Tensor, lengths: torch.Tensor, method: str) -> torch.Tensor:
    """FP32 flat pooling, one final normalization; inactive slots have zero weight.

    Input integrity is checked once by the caller (``validate_children``), not
    inside timed inference. Cancellation is checked on the generated outputs.
    """
    if method not in BASELINES:
        raise ValueError(f"unknown baseline: {method}")
    if method == "first_chunk":
        return F.normalize(children[:, 0].float(), dim=-1)
    x = F.normalize(children.float(), dim=-1)
    weights = (lengths > 0).to(x) if method == "unweighted_mean" else lengths.to(x)
    # Dividing by total mass matches the model's structural prior and avoids
    # needlessly large sums. A common scale does not affect final normalization.
    weights = weights / weights.sum(1, keepdim=True)
    return F.normalize((x * weights[..., None]).sum(1), dim=-1)


def validate_vectors(values, name: str = "vectors", unit: bool = False) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name}: expected finite rank-2 vectors")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if (norms <= 1e-12).any():
        raise ValueError(f"{name}: zero vector (including possible pooling cancellation)")
    if unit and not np.allclose(norms, 1, atol=2e-5, rtol=0):
        raise ValueError(f"{name}: vectors are not unit normalized")
    return values


def cosine_rows(values, targets) -> np.ndarray:
    values = validate_vectors(values).astype(np.float64)
    targets = validate_vectors(targets, "targets").astype(np.float64)
    if values.shape != targets.shape:
        raise ValueError("paired vectors must have equal shapes")
    return (values * targets).sum(1) / (
        np.linalg.norm(values, axis=1) * np.linalg.norm(targets, axis=1)
    )


def shared_bootstrap_means(columns: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """Resample documents once for all columns; never resample checkpoints as rows."""
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim != 2 or not len(columns) or not np.isfinite(columns).all():
        raise ValueError("bootstrap expects a finite [documents, statistics] matrix")
    rng = np.random.default_rng(seed)
    output = np.empty((reps, columns.shape[1]), dtype=np.float64)
    for index in range(reps):
        output[index] = columns[rng.integers(0, len(columns), size=len(columns))].mean(0)
    return output


def summarize_scores(values, bootstrap_means=None, difference=False) -> dict:
    values = np.asarray(values, dtype=np.float64)
    summary = {
        "documents": len(values), "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {str(q): float(np.quantile(values, q)) for q in (0, .05, .5, .95, 1)},
    }
    if bootstrap_means is not None:
        summary["ci95"] = np.percentile(bootstrap_means, [2.5, 97.5]).tolist()
    if difference:
        summary["fraction_positive"] = float((values > 0).mean())
        summary["fraction_zero"] = float((values == 0).mean())
    return summary


def fidelity_summary(scores: dict[str, np.ndarray], mask: np.ndarray, reps: int, seed: int) -> dict:
    """Summarize method scores and paired deltas using exactly shared resamples."""
    values = {key: np.asarray(value)[mask] for key, value in scores.items()}
    differences = {key: values[a] - values[b] for key, (a, b) in CONTRASTS.items()}
    combined = {**values, **differences}
    samples = shared_bootstrap_means(np.column_stack(list(combined.values())), reps, seed)
    result = {key: summarize_scores(value, samples[:, index], key in differences)
              for index, (key, value) in enumerate(combined.items())}
    return {"methods": {key: result[key] for key in values},
            "contrasts": {key: result[key] for key in differences}}


def teacher_permutations(labels, reps: int, seed: int) -> np.ndarray:
    """Same-corpus derangements: no correct document pairing survives."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    result = np.tile(np.arange(len(labels)), (reps, 1))
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("derangements require at least two documents per corpus")
        for row in result:
            while True:
                permuted = rng.permutation(indices)
                if np.all(permuted != indices):
                    row[indices] = permuted
                    break
    return result


def shuffled_cosines(values, targets, permutations: np.ndarray) -> np.ndarray:
    values = validate_vectors(values).astype(np.float64)
    targets = validate_vectors(targets).astype(np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    return np.stack([np.einsum("ij,ij->i", values, targets[p]) for p in permutations])


def benchmark(call: Callable, batches: list[tuple], device: torch.device,
              warmup: int, repeats: int, documents: int) -> dict:
    """Wall time including Python launches; excludes input/output device transfers."""
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run():
        for args in batches:
            call(*args)

    with torch.inference_mode():
        for _ in range(warmup):
            run()
        synchronize()
        seconds = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            synchronize()
            seconds.append(time.perf_counter() - start)
    median = float(np.median(seconds))
    return {"pass_seconds": seconds, "median_seconds": median,
            "median_ms_per_document": median * 1000 / documents,
            "median_documents_per_second": documents / median,
            "documents": documents, "warmup_passes": warmup, "measured_passes": repeats}
