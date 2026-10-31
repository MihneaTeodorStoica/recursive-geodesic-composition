from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from recursive_geodesic_composition.architecture import GeodesicMerge, weighted_average
from recursive_geodesic_composition.recursive_reuse import (
    binary_reuse_four, k4_accounting, population_summary, summarize_tree_scores, validate_four,
)
from recursive_geodesic_composition.safe_baselines import baseline_parent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_recursive_reuse import independent_binary_tree
from evaluate_recursive_reuse import load_protocol, select_exact_four


class RecordingMerger:
    arity = 4

    def __init__(self):
        self.calls = []

    def merge(self, children, lengths):
        self.calls.append((children.clone(), lengths.clone()))
        return weighted_average(children, lengths)


def test_binary_reuse_preserves_order_same_operator_and_exact_counts():
    children = F.normalize(torch.arange(1, 3 * 4 * 5 + 1, dtype=torch.float32).reshape(3, 4, 5), dim=-1)
    lengths = torch.tensor([[1024., 1024., 1024., 1.], [1024., 1024., 1024., 17.], [1024., 1024., 1024., 1024.]])
    model = RecordingMerger()
    root, parents, masses = binary_reuse_four(model, children, lengths, return_intermediates=True)
    assert len(model.calls) == 2
    first_x, first_n = model.calls[0]
    assert first_x.shape == (6, 4, 5)
    assert torch.equal(first_x[:, :2], children.reshape(6, 2, 5))
    assert torch.equal(first_n[:, :2], lengths.reshape(6, 2))
    assert not first_x[:, 2:].any() and not first_n[:, 2:].any()
    root_x, root_n = model.calls[1]
    assert torch.equal(root_x[:, :2], parents)
    assert torch.equal(root_n[:, :2], masses)
    assert torch.equal(masses, torch.tensor([[2048., 1025.], [2048., 1041.], [2048., 2048.]]))
    assert torch.equal(root_n.sum(1), lengths.sum(1))
    assert torch.allclose(root.norm(dim=-1), torch.ones(3))


def test_batched_tree_matches_independent_three_call_implementation():
    torch.manual_seed(72)
    model = GeodesicMerge(d=8, hidden=16, layers=1, heads=2, dropout=0., arity=4,
                          importance_mode="semantic", geodesic_parameterization="tangent_expmap").eval()
    # Exercise a nonzero correction and nonuniform salience, not only initialization.
    with torch.no_grad():
        model.direction.weight.normal_(0, .01)
        model.salience[-1].weight.normal_(0, .01)
    x = F.normalize(torch.randn(7, 4, 8), dim=-1)
    n = torch.tensor([[1024., 1024., 1024., 21. + i] for i in range(7)])
    validate_four(x, n)
    with torch.inference_mode():
        actual = binary_reuse_four(model, x, n, return_intermediates=True)
        expected = independent_binary_tree(model, x, n)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=0)


def test_recursively_normalized_pooling_is_not_flat_pooling():
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [1., 0.]]])
    n = torch.ones(1, 4)
    recursive = binary_reuse_four(RecordingMerger(), x, n)
    flat = baseline_parent(x, n, "token_weighted_mean")
    expected_flat = F.normalize(x.sum(1), dim=-1)
    assert torch.allclose(flat, expected_flat)
    assert not torch.allclose(recursive, flat, atol=1e-3)


def test_exact_four_validation_rejects_partial_and_invalid_inputs():
    x = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match="exactly four"):
        validate_four(x, torch.tensor([[1., 1., 1., 0.]]))
    with pytest.raises(ValueError, match="nonfinite"):
        validate_four(x * float("nan"), torch.ones(1, 4))
    with pytest.raises(ValueError, match="expected"):
        binary_reuse_four(RecordingMerger(), x[:, :3], torch.ones(1, 3))


@pytest.mark.parametrize("tokens,expected", [(1024, (1, 0, 0)), (1025, (2, 1, 1)),
                                            (4096, (4, 1, 1)), (4097, (5, 2, 2)),
                                            (8192, (8, 2, 3)), (8193, (9, 2, 3)),
                                            (16384, (16, 2, 5)), (16385, (17, 3, 6)),
                                            (58731, (58, 3, 20))])
def test_population_accounting_handles_singleton_carry(tokens, expected):
    assert k4_accounting(tokens) == expected


def test_population_strict_context_boundary_and_bins():
    values = [1025, 2048, 2049, 4096, 4097, 8192, 8193, 16384, 16385, 32768, 32769, 58731]
    rows = [{"id": str(index), "full_token_count": value} for index, value in enumerate(values)]
    stats = population_summary(rows, [2048, 4096, 8192, 16384, 32768, None], 8192)
    assert stats["documents_strictly_above_context"] == 6
    assert stats["documents_at_context_threshold"] == 1
    assert [entry["documents"] for entry in stats["length_bins"]] == [2] * 6
    assert stats["depth_histogram"] == {"1": 4, "2": 4, "3": 4}
    with pytest.raises(ValueError, match="unique"):
        population_summary(rows + rows[:1], [2048, None], 8192)


def test_tree_statistics_use_paired_differences_with_correct_sign():
    direct = np.array([.97, .98, .99, .95])
    scores = {"direct_four": direct, "recursive_binary": direct - .002,
              "flat_token_weighted": direct - .005, "flat_unweighted": direct - .01,
              "direct_recursive_output_cosine": np.array([.998] * 4)}
    result = summarize_tree_scores(scores, np.ones(4, dtype=bool), 200, 12, [.001, .005, .01])
    delta = result["contrasts"]["recursive_minus_direct"]
    assert delta["mean"] == pytest.approx(-.002)
    assert delta["ci95"] == pytest.approx([-.002, -.002])
    loss = result["degradation_direct_minus_recursive"]
    assert loss["mean"] == pytest.approx(.002)
    assert loss["fraction_degraded"] == 1
    assert loss["fraction_loss_above"] == {"0.001": 1., "0.005": 0., "0.01": 0.}
    assert result == summarize_tree_scores(scores, np.ones(4, dtype=bool), 200, 12, [.001, .005, .01])


def test_reuse_protocol_keeps_preceding_studies_immutable():
    cfg, baseline, official, published = load_protocol()
    assert cfg["expected_documents"] == 606
    assert cfg["merger_updates"] == cfg["teacher_encoding_calls"] == cfg["classifier_fits"] == 0
    assert cfg["output_dir"] not in (baseline["output_dir"], official["output_dir"])
    assert cfg["seeds"] == published["seed_order"]
    assert cfg["primary_contrast"] == "recursive_minus_direct_teacher_cosine"


def test_selection_matches_frozen_exact_four_document_identities_without_tensor_cache():
    import json
    metadata = json.loads((ROOT / "results/safe_baselines_v1/DOCUMENT_METADATA.json").read_text())
    rows = [row | {"lengths": torch.ones(row["active_children"])} for row in metadata]
    cfg, _, _, published = load_protocol()
    selected, indices = select_exact_four(rows, cfg, published)
    assert len(selected) == 606
    assert np.array_equal(indices, np.flatnonzero([row["active_children"] == 4 for row in metadata]))
    rows[indices[0]]["id"] = "wrong-document"
    with pytest.raises(ValueError, match="identity/order"):
        select_exact_four(rows, cfg, published)
