"""Synthetic tests of document-paired AUC and continuation contrasts."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from recursive_geodesic_composition.continuation_comparison import DocumentWeightedAUC, paired_auc, paired_summary


def fixture_bundle(scores=(.1, .3, .3, .9, .3, .2, .6, .6)):
    return {"y": [1, 1, 1, 1, 0, 0, 0, 0], "scores": list(scores), "groups": ["a", "b", "c", "d"] * 2}


def test_weighted_auc_handles_ties_repetitions_and_zero_weight_documents():
    bundle = fixture_bundle()
    evaluator = DocumentWeightedAUC(bundle)
    rng = np.random.default_rng(72)
    for _ in range(100):
        counts = np.bincount(rng.integers(0, 4, 4), minlength=4)
        reference = roc_auc_score(bundle["y"], bundle["scores"], sample_weight=np.tile(counts, 2))
        assert evaluator.raw(counts) == pytest.approx(reference, abs=1e-15)
        assert evaluator.safe(counts) == pytest.approx(max(reference, 1-reference), abs=1e-15)


def test_auc_orientation_invariance_and_all_ties():
    a = DocumentWeightedAUC(fixture_bundle())
    b = DocumentWeightedAUC(fixture_bundle(-np.asarray(fixture_bundle()["scores"])))
    for counts in ([1, 1, 1, 1], [4, 0, 0, 0], [1, 0, 2, 1]):
        assert a.safe(counts) == pytest.approx(b.safe(counts))
    assert DocumentWeightedAUC(fixture_bundle([.5] * 8)).safe([2, 0, 1, 1]) == .5


@pytest.mark.parametrize("field,value", [("y", [1] * 8), ("scores", [np.nan] * 8), ("groups", ["a"] * 8), ("scores", [1])])
def test_invalid_bundles_rejected(field, value):
    with pytest.raises(ValueError):
        DocumentWeightedAUC(fixture_bundle() | {field: value})


@pytest.mark.parametrize("counts", [[1, 2], [0, 0, 0, 0], [1, -1, 2, 2], [np.nan, 1, 1, 1]])
def test_invalid_weights_rejected(counts):
    with pytest.raises(ValueError):
        DocumentWeightedAUC(fixture_bundle()).raw(counts)


def test_auc_row_order_is_irrelevant_but_document_pairing_is_preserved():
    bundle = fixture_bundle()
    order = [7, 0, 3, 1, 6, 5, 4, 2]
    shuffled = {k: [v[i] for i in order] for k, v in bundle.items()}
    result = paired_auc([bundle], [shuffled], 100, 7)
    assert result["three_seed_mean"]["final_minus_source"] == {"mean": 0., "ci95": [0., 0.]}


def test_paired_auc_averages_checkpoint_aucs_and_reproduces_explicit_document_bootstrap():
    a = fixture_bundle()
    b = fixture_bundle([.9, .8, .6, .7, .1, .2, .3, .4])
    c = fixture_bundle([.3, .2, .1, .7, .4, .6, .6, .5])
    result = paired_auc([a, c], [b, a], 100, 91)
    raw = lambda bundle, w: roc_auc_score(bundle["y"], bundle["scores"], sample_weight=np.tile(w, 2))
    safe = lambda bundle, w: max(raw(bundle, w), 1-raw(bundle, w))
    rng = np.random.default_rng(91)
    samples = []
    for _ in range(100):
        w = np.bincount(rng.integers(0, 4, 4), minlength=4)
        samples.append((safe(b, w) + safe(a, w) - safe(a, w) - safe(c, w)) / 2)
    row = result["three_seed_mean"]["final_minus_source"]
    np.testing.assert_allclose(row["ci95"], np.percentile(samples, [2.5, 97.5]), atol=1e-15, rtol=0)
    assert row["mean"] == pytest.approx((safe(b, np.ones(4))-safe(c, np.ones(4)))/2)
    averaged_scores = np.mean([a["scores"], c["scores"]], axis=0)
    wrong = DocumentWeightedAUC(fixture_bundle(averaged_scores)).safe(np.ones(4))
    assert result["three_seed_mean"]["source"]["mean"] != wrong


def test_mismatched_classifier_holdout_documents_rejected():
    changed = fixture_bundle() | {"groups": ["e", "b", "c", "d"] * 2}
    with pytest.raises(ValueError, match="holdout documents differ"):
        paired_auc([fixture_bundle()], [changed], 20, 3)


def test_scalar_delta_interval_preserves_document_pairing():
    source = np.array([.1, .4, .3, .7])
    final = source + .02
    result = paired_summary({"source": source, "final": final, "delta": final-source}, np.ones(4, bool), 100, 3)
    np.testing.assert_allclose(result["delta"]["ci95"], [.02, .02])
    assert result["delta"]["fraction_positive"] == 1
