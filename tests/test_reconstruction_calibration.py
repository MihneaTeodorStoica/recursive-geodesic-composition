from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from recursive_geodesic_composition.reconstruction_calibration import (
    METHODS, PERMUTATIONS, DERANGEMENTS, cross_cosines, assignment_scores,
    geometry_calibration, reconstruction_summary,
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_reconstruction_calibration import load_protocol, predict, score_matrices


def test_permutations_are_exhaustive_and_include_nine_derangements():
    assert len(PERMUTATIONS) == len({tuple(p) for p in PERMUTATIONS}) == 24
    assert tuple(PERMUTATIONS[0]) == (0, 1, 2, 3)
    assert DERANGEMENTS.sum() == 9
    assert all(all(p[i] != i for i in range(4)) for p in PERMUTATIONS[DERANGEMENTS])


def test_perfect_order_and_reversed_order_are_distinguished():
    x = np.eye(4)[None]
    correct = assignment_scores(cross_cosines(x, x))
    reversed_scores = assignment_scores(cross_cosines(x[:, ::-1], x))
    assert correct["ordered"] == 1
    assert correct["shuffled_all24"] == .25
    assert correct["shuffled_deranged9"] == 0
    assert correct["ordered_minus_shuffled"] == .75
    assert reversed_scores["ordered"] == 0
    assert reversed_scores["ordered_minus_shuffled"] == -.25
    assert reversed_scores["oracle_best_assignment"] == 1


def test_copied_parent_has_no_order_advantage_for_any_permutation():
    rng = np.random.default_rng(45)
    x, parents = rng.normal(size=(17, 4, 8)), rng.normal(size=(17, 8))
    predictions = np.repeat(parents[:, None, :], 4, axis=1)
    cross = cross_cosines(predictions, x)
    scores = assignment_scores(cross)
    assert np.array_equal(scores["ordered_minus_shuffled"], np.zeros(17))
    assert np.array_equal(scores["ordered_minus_deranged"], np.zeros(17))
    assert np.array_equal(scores["oracle_best_minus_ordered"], np.zeros(17))
    for permutation in itertools.permutations(range(4)):
        np.testing.assert_allclose(cross[:, np.arange(4), permutation].mean(1), scores["ordered"], atol=1e-15)


def test_shuffle_algebra_and_gain_relation():
    rng = np.random.default_rng(62)
    c = rng.uniform(-1, 1, size=(10, 4, 4))
    s = assignment_scores(c)
    diagonal = np.trace(c, axis1=1, axis2=2) / 4
    off = (c.sum((1, 2)) - 4 * diagonal) / 12
    np.testing.assert_allclose(s["shuffled_all24"], c.mean((1, 2)), atol=1e-15)
    np.testing.assert_allclose(s["shuffled_deranged9"], off, atol=1e-15)
    np.testing.assert_allclose(s["ordered_minus_deranged"], s["ordered_minus_shuffled"] * 4 / 3, atol=1e-15)


def test_unweighted_copy_is_optimal_only_among_shared_vector_outputs():
    x = np.eye(4)[None]
    parent = np.ones((1, 4)) / 2
    copied = assignment_scores(cross_cosines(np.repeat(parent[:, None], 4, axis=1), x))["ordered"]
    np.testing.assert_allclose(copied, np.linalg.norm(x.mean(1), axis=-1))
    assert copied == .5
    assert assignment_scores(cross_cosines(x, x))["ordered"] > copied


@pytest.mark.parametrize("bad", [np.zeros((2, 3, 4)), np.full((2, 4, 4), np.nan), np.full((2, 4, 4), 1.1)])
def test_invalid_cross_matrices_rejected(bad):
    with pytest.raises(ValueError):
        assignment_scores(bad)


def test_zero_predictions_are_not_allowed():
    with pytest.raises(ValueError, match="zero"):
        cross_cosines(np.zeros((2, 4, 8)), np.ones((2, 4, 8)))


def test_geometry_cap_orthogonal_targets_and_identical_targets():
    zero = geometry_calibration(np.zeros((2, 4)))
    np.testing.assert_allclose(zero["loose_cap_cosine_ceiling"], .25, atol=1e-15)
    assert np.all(zero["fraction_children_outside_cap"] == 1)
    same = geometry_calibration(np.ones((2, 4)))
    assert np.all(same["loose_cap_cosine_ceiling"] == 1)
    assert np.all(same["fraction_children_outside_cap"] == 0)


def test_predictor_receives_only_parent_not_labels_lengths_or_masks():
    parents = F.normalize(torch.randn(7, 8), dim=-1)
    calls = []
    def spy(*args, **kwargs):
        assert len(args) == 1 and not kwargs
        assert args[0].ndim == 2 and args[0].shape[1] == 8
        calls.append(len(args[0]))
        return args[0][:, None, :].expand(-1, 4, -1)
    result = predict(spy, parents, torch.device("cpu"), 3)
    assert calls == [3, 3, 1]
    torch.testing.assert_close(result[:, 0], parents)


def test_summary_has_paired_constant_gain_and_document_unit():
    copy = np.full((3, 4, 4), .7)
    better = copy.copy()
    better[:, np.arange(4), np.arange(4)] += .1
    scores = {method: assignment_scores(better if method == "strict_decoder" else copy) for method in METHODS}
    s = reconstruction_summary(scores, np.ones(3, dtype=bool), 100, 83)
    delta = s["contrasts"]["strict_minus_parent_copy"]
    assert delta["documents"] == 3
    assert delta["mean"] == pytest.approx(.1)
    assert delta["ci95"] == pytest.approx([.1, .1])
    assert s["methods"]["strict_decoder"]["ordered_minus_shuffled"]["mean"] == pytest.approx(.075)


def test_checkpoint_scores_are_averaged_after_each_assignment_diagnostic():
    cfg, *_ = load_protocol()
    matrices = {}
    for seed, factor in zip(cfg["seeds"], (.9, .8, .7)):
        for method in ("parent_copy", "strict_decoder", "frozen_probe"):
            matrices[f"{method}_seed{seed}"] = np.eye(4)[None] * factor
    matrices.update({key: np.full((1, 4, 4), .5) for key in ("weighted_copy", "unweighted_copy")})
    result = score_matrices(matrices, cfg)
    assert result["three_seed_mean"]["strict_decoder"]["ordered"][0] == pytest.approx(.8)
    assert result["three_seed_mean"]["strict_decoder"]["ordered_minus_shuffled"][0] == pytest.approx(.6)


def test_protocol_is_inference_only_and_separate_from_prior_results():
    cfg, tree, baseline, _, _ = load_protocol()
    assert cfg["output_dir"] not in (tree["output_dir"], baseline["output_dir"])
    assert cfg["expected_documents"] == 606
    assert cfg["probe_updates"] == cfg["decoder_updates"] == cfg["merger_updates"] == 0
    assert cfg["shuffle"]["monte_carlo_draws"] == 0
