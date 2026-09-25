"""Pure algorithms for additive order, tree, probe and retrieval controls."""
from __future__ import annotations
import itertools
import numpy as np
import torch
import torch.nn.functional as F

PERMUTATIONS = list(itertools.permutations(range(4)))


def binary_trees(indices=(0, 1, 2, 3)):
    if len(indices) == 1:
        return [indices[0]]
    return [(left, right) for i in range(1, len(indices))
            for left in binary_trees(indices[:i]) for right in binary_trees(indices[i:])]


def binary_tree(model, children, masses, tree):
    if isinstance(tree, int):
        return children[:, tree], masses[:, tree]
    left, a = binary_tree(model, children, masses, tree[0])
    right, b = binary_tree(model, children, masses, tree[1])
    x, n = torch.stack((left, right), 1), torch.stack((a, b), 1)
    return model.merge(F.pad(x, (0, 0, 0, 2)), F.pad(n, (0, 2))), a + b


def ragged_reduce(model, leaves, masses, arity=4, batch_size=64):
    """Level-packed contiguous reduction; carry singletons, preserve true mass."""
    if arity not in (2, 4) or len(leaves) != len(masses) or not leaves:
        raise ValueError("invalid reduction inputs")
    nodes, counts = list(leaves), list(masses)
    total = [n.sum().item() for n in counts]
    depths, merges = [0] * len(nodes), [0] * len(nodes)
    while any(len(x) > 1 for x in nodes):
        jobs, positions, next_x, next_n = [], [], [[] for _ in nodes], [[] for _ in nodes]
        for d, (x, n) in enumerate(zip(nodes, counts)):
            if len(x) > 1:
                depths[d] += 1
            for start in range(0, len(x), arity):
                child, mass = x[start:start+arity], n[start:start+arity]
                pos = len(next_x[d])
                next_n[d].append(mass.sum())
                if len(child) == 1:
                    next_x[d].append(child[0])
                else:
                    merges[d] += 1
                    next_x[d].append(None)
                    jobs.append((F.pad(child, (0, 0, 0, 4-len(child))), F.pad(mass, (0, 4-len(mass)))))
                    positions.append((d, pos))
        for start in range(0, len(jobs), batch_size):
            part = jobs[start:start+batch_size]
            output = model.merge(torch.stack([p[0] for p in part]), torch.stack([p[1] for p in part]))
            for value, (d, pos) in zip(output, positions[start:start+batch_size]):
                next_x[d][pos] = value
        nodes = [torch.stack(x) for x in next_x]
        counts = [torch.stack(n) for n in next_n]
    if [n.item() for n in counts] != total:
        raise AssertionError("token mass changed")
    return torch.stack([x[0] for x in nodes]), {"depths": depths, "merges": merges, "total_masses": total}


def slot_scores(predictions, targets):
    """Ordered and all-assignment scores, accepting arbitrary leading axes."""
    p = F.normalize(predictions.double(), dim=-1)
    x = F.normalize(targets.double(), dim=-1)
    ordered = (p * x).sum(-1).mean(-1)
    shuffled = (p.mean(-2) * x.mean(-2)).sum(-1)
    return ordered, shuffled


class FlexibleProbe(torch.nn.Module):
    def __init__(self, dimension=768, hidden=512, family="bounded", centroid=None):
        super().__init__()
        if family not in {"bounded", "unbounded", "direct"}:
            raise ValueError("unknown probe family")
        self.dimension, self.family = dimension, family
        self.network = torch.nn.Sequential(torch.nn.LayerNorm(dimension), torch.nn.Linear(dimension, hidden),
            torch.nn.GELU(), torch.nn.Linear(hidden, 4*dimension))
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)
        if family == "direct":
            if centroid is None or centroid.shape != (4, dimension) or not torch.all(centroid.norm(dim=-1) > 0):
                raise ValueError("direct probe needs a nonzero TRAIN slot centroid")
            with torch.no_grad():
                self.network[-1].bias.copy_(centroid.reshape(-1))

    def forward(self, parent):
        p = F.normalize(parent, dim=-1)
        value = self.network(p).reshape(-1, 4, self.dimension)
        if self.family == "bounded":
            value = p[:, None] + .25 * torch.tanh(value) / self.dimension ** .5
        elif self.family == "unbounded":
            value = p[:, None] + .25 * value / self.dimension ** .5
        return F.normalize(value, dim=-1)


def retrieval_metrics(scores, relevance):
    """Stable ties use the input's document-ID-sorted column order."""
    scores, relevance = np.asarray(scores), np.asarray(relevance)
    if scores.shape != relevance.shape or scores.ndim != 2 or not np.isfinite(scores).all() or np.any(relevance < 0) or np.any((relevance > 0).sum(1) == 0):
        raise ValueError("invalid retrieval score/relevance matrices")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    rel = np.take_along_axis(relevance, order, axis=1)
    ideal = np.sort(relevance, axis=1)[:, ::-1][:, :10]
    discount = 1 / np.log2(np.arange(rel.shape[1]) + 2)
    positive = rel > 0
    n = (relevance > 0).sum(1)
    return {"ndcg_at_10": ((2.**rel-1)*discount).sum(1) / ((2.**ideal-1)*discount).sum(1),
            "recall_at_1": positive[:, 0]/n, "recall_at_10": positive.sum(1)/n,
            "mrr_at_10": (positive / (np.arange(rel.shape[1])+1)).max(1)}


def positive_components(relevance):
    """Queries sharing ANY positive document belong to the same cluster."""
    rel = np.asarray(relevance) > 0
    parent = list(range(len(rel)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for col in rel.T:
        idx = np.flatnonzero(col)
        for i in idx[1:]:
            parent[find(int(i))] = find(int(idx[0]))
    return np.array([find(i) for i in range(len(rel))])


def cluster_bootstrap(values, groups, reps, seed):
    """Query-weighted mean; sample positive-document clusters, not queries."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    _, inv = np.unique(groups, return_inverse=True)
    k = int(inv.max())+1
    sums = np.zeros((k, values.shape[1]))
    np.add.at(sums, inv, values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        sample = rng.integers(0, k, k)
        out.append(sums[sample].sum(0) / counts[sample].sum())
    return np.asarray(out)
