from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from recursive_geodesic_composition.architecture import GeodesicMerge
from recursive_geodesic_composition.safe_baselines import (
    BASELINES, baseline_parent, cosine_rows, fidelity_summary, shared_bootstrap_means,
    teacher_permutations, validate_children, validate_vectors,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from safe_baselines_protocol import load_protocol, stable_hash, validate_population


@pytest.mark.parametrize("arity", (2, 3, 4))
def test_masked_baselines_are_flat_and_normalize_children(arity):
    torch.manual_seed(31)
    children = torch.randn(3, 4, 8)
    lengths = torch.zeros(3, 4)
    lengths[:, :arity] = torch.arange(1, arity + 1)
    validate_children(children, lengths)
    x = F.normalize(children[:, :arity], dim=-1)
    expected = {
        "first_chunk": x[:, 0],
        "unweighted_mean": F.normalize(x.sum(1), dim=-1),
        "token_weighted_mean": F.normalize((x * lengths[:, :arity, None]).sum(1), dim=-1),
    }
    for name in BASELINES:
        actual = baseline_parent(children, lengths, name)
        assert torch.allclose(actual, expected[name], atol=1e-6)
        assert torch.allclose(actual.norm(dim=-1), torch.ones(3), atol=1e-6)
        # Masked trailing data must not affect any result.
        altered = children.clone()
        altered[:, arity:] = 1000
        assert torch.allclose(actual, baseline_parent(altered, lengths, name))


def test_nonfinite_zero_and_nonprefix_children_are_rejected():
    children = torch.ones(1, 4, 8)
    with pytest.raises(ValueError, match="contiguous"):
        validate_children(children, torch.tensor([[1., 0., 1., 0.]]))
    with pytest.raises(ValueError, match="at least two"):
        validate_children(children, torch.tensor([[1., 0., 0., 0.]]))
    with pytest.raises(ValueError, match="zero-norm"):
        validate_children(children * 0, torch.ones(1, 4))
    with pytest.raises(ValueError, match="nonfinite"):
        validate_children(children * float("nan"), torch.ones(1, 4))
    with pytest.raises(ValueError, match="zero vector"):
        validate_vectors(np.zeros((2, 8)))


def test_pooling_cancellation_is_rejected_at_output_validation():
    children = torch.tensor([[[1., 0.], [-1., 0.]]])
    output = baseline_parent(children, torch.ones(1, 2), "unweighted_mean")
    with pytest.raises(ValueError, match="zero vector"):
        validate_vectors(output.numpy(), unit=True)


def test_token_rescaling_leaves_pooling_unchanged():
    children = torch.randn(2, 4, 8)
    lengths = torch.tensor([[5., 4., 0., 0.], [2., 3., 4., 5.]])
    for name in BASELINES:
        assert torch.allclose(baseline_parent(children, lengths, name),
                              baseline_parent(children, lengths * 17, name), atol=1e-6)


def test_reference_is_the_actual_model_anchor_and_does_not_change_parent():
    torch.manual_seed(15)
    model = GeodesicMerge(d=8, hidden=16, layers=1, heads=2, dropout=0.,
                          arity=4, importance_mode="semantic", geodesic_parameterization="tangent_expmap").eval()
    children = torch.randn(3, 4, 8)
    counts = torch.tensor([[4., 1., 0., 0.], [4., 4., 1., 0.], [4., 4., 4., 1.]])
    with torch.inference_mode():
        parent, aux = model.merge(children, counts, return_aux=True)
        plain = model.merge(children, counts)
    assert torch.equal(parent, plain)
    expected = F.normalize((F.normalize(children, dim=-1) * aux["effective_weights"][..., None]).sum(1), dim=-1)
    assert torch.equal(aux["anchor"], expected)
    # At zero correction initialization the parent equals the anchor.
    assert torch.equal(parent, aux["anchor"])


def test_cosine_and_bootstrap_are_paired_deterministic():
    x = np.array([[1., 0.], [0., 2.]])
    assert np.allclose(cosine_rows(x, x), [1, 1])
    columns = np.array([[1., 3.], [2., 4.], [3., 5.]])
    samples = shared_bootstrap_means(columns, 100, 17)
    assert np.array_equal(samples, shared_bootstrap_means(columns, 100, 17))
    # Shared resampling preserves a constant per-document contrast exactly.
    assert np.allclose(samples[:, 1] - samples[:, 0], 2)


def test_fidelity_scores_use_paired_differences_and_document_count():
    scores = {name: np.array([.5, .6, .7]) for name in (*BASELINES, "learned_reference", "learned_parent")}
    scores["learned_parent"] += .02
    summary = fidelity_summary(scores, np.ones(3, dtype=bool), 100, 17)
    delta = summary["contrasts"]["parent_minus_weighted"]
    assert delta["documents"] == 3
    assert delta["mean"] == pytest.approx(.02)
    assert delta["ci95"] == pytest.approx([.02, .02])
    assert delta["fraction_positive"] == 1


def test_calibration_is_same_corpus_and_has_no_self_pairings():
    labels = np.array(["a"] * 5 + ["b"] * 7)
    permutations = teacher_permutations(labels, 100, 91)
    assert np.array_equal(permutations, teacher_permutations(labels, 100, 91))
    for permutation in permutations:
        assert np.all(permutation != np.arange(len(labels)))
        assert np.array_equal(labels[permutation], labels)
        assert np.array_equal(np.sort(permutation), np.arange(len(labels)))


def test_protocol_is_additive_and_frozen_inputs_still_match():
    cfg, official, published = load_protocol()
    assert cfg["merger_updates"] == 0
    assert cfg["cache_dir"] != official["cache_dir"]
    assert cfg["output_dir"] != official["output_dir"]
    assert cfg["seeds"] == published["seed_order"]
    assert cfg["analysis_role"].startswith("additive_post_review")


def test_population_validation_rejects_order_drift():
    rows = [{"id": f"a:{162000 + i}", "source_index": 162000 + i, "corpus": "a",
             "selected_rank": i, "safe_prefix_token_count": 1025} for i in range(1000)]
    published = {"cache_provenance": {"corpora": {"a": {
        "document_order_sha256": stable_hash([row["id"] for row in rows]),
        "safe_prefix_tokens": {"sum": 1025000},
    }}}}
    validate_population(rows, "a", published)
    with pytest.raises(ValueError, match="document order"):
        validate_population(rows[::-1], "a", published)


def test_classifier_split_is_identical_across_methods_and_keeps_pairs():
    sys.path.insert(0, str(ROOT / "scripts/official_v6"))
    from evaluate_official_test_v6_recovery import grouped_attack_split
    rng = np.random.default_rng(77)
    real, fake = rng.normal(size=(40, 8)), rng.normal(size=(40, 8))
    groups = np.array([f"doc:{i}" for i in range(40)])
    cfg = {"test_size": .5, "split_seed": 1729}
    _, y, paired, train, held = grouped_attack_split(real, fake, groups, cfg)
    _, _, _, train2, held2 = grouped_attack_split(real, fake * 2, groups, cfg)
    assert np.array_equal(train, train2) and np.array_equal(held, held2)
    assert not set(paired[train]) & set(paired[held])
    for doc in groups:
        indices = np.flatnonzero(paired == doc)
        assert set(y[indices]) == {0, 1}
        assert all(index in train for index in indices) or all(index in held for index in indices)
