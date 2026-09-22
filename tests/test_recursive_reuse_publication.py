"""Committed recursive-reuse artifacts are testable without local input tensors."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from recursive_geodesic_composition.recursive_reuse import CONTRASTS, summarize_tree_scores

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/recursive_reuse_v1"
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_recursive_reuse import characterize_population, load_protocol


def read(name):
    return json.loads((OUTPUT / name).read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_recursive_artifact_hashes_frozen_sources_and_complete_audit():
    for name, expected in read("ARTIFACT_HASHES.json").items():
        path = OUTPUT / name
        if name == "vectors.pt" and not path.exists():
            continue
        assert digest(path) == expected, name
    result, audit, manifest = read("RESULTS.json"), read("AUDIT.json"), read("INFERENCE_MANIFEST.json")
    assert audit["passed"] and all(audit["checks"].values())
    assert audit["documents"] == 606 and audit["checkpoints"] == 3
    assert audit["results_sha256"] == digest(OUTPUT / "RESULTS.json")
    for section in ("source_sha256", "immutable_input_sha256"):
        assert result[section] == manifest[section]
        for path, expected in result[section].items():
            assert digest(ROOT / path) == expected, path
    assert manifest["new_teacher_encoding_calls"] == manifest["merger_updates"] == manifest["classifier_fits"] == 0
    for check in manifest["reference_checks"].values():
        if "direct_max_abs_difference_from_previous_vectors" in check:
            assert check["direct_max_abs_difference_from_previous_vectors"] == 0
        else:
            assert check["max_abs_difference_from_previous_vectors"] == 0
    for seed in audit["independent_replay_max_abs_differences"].values():
        assert max(seed.values()) <= result["configuration"]["independent_tree_max_absolute_tolerance"]


def test_recursive_table_and_paper_match_computed_results():
    result = read("RESULTS.json")
    pooled = result["summaries"]["three_seed_mean"]["pooled"]
    table = (OUTPUT / "recursive_reuse_table.tex").read_text()
    names = {"flat_unweighted": "Flat unweighted pooling", "flat_token_weighted": "Flat token-weighted pooling",
             "direct_four": "Direct four-child merge", "recursive_binary": "Recursive binary tree"}
    for method, label in names.items():
        value = pooled["metrics"][method]["mean"]
        delta = value - pooled["metrics"]["direct_four"]["mean"]
        cost = result["timing"][method]["mean_checkpoint_median_ms_per_document"]
        assert f"{label} & {value:.4f} & {delta:+.6f} & {cost:.4f}" in table
    paper = (ROOT / "paper/main.tex").read_text()
    for target in ("recursive_reuse_table.tex", "full_population_table.tex", "degradation_ecdf.png"):
        assert f"../results/recursive_reuse_v1/{target}" in paper
    primary = pooled["contrasts"]["recursive_minus_direct"]
    assert f"{primary['mean']:.6f}" in paper
    assert f"[{primary['ci95'][0]:.6f},{primary['ci95'][1]:.6f}]" in paper
    for seed in [*map(str, result["checkpoint_order"]), "three_seed_mean"]:
        cell = result["summaries"][seed]["pooled"]
        for method in ("direct_four", "recursive_binary"):
            assert f"{cell['metrics'][method]['mean']:.6f}" in paper
        assert f"{cell['degradation_direct_minus_recursive']['quantiles']['1']:.6f}" in paper


def test_all_paired_scores_intervals_and_degradation_reproduce_from_published_points():
    result, rows = read("RESULTS.json"), read("DOCUMENT_SCORES.json")
    cfg = result["configuration"]
    assert len(rows) == len({row["id"] for row in rows}) == 606
    names = cfg["methods"] + ["direct_recursive_output_cosine"]
    for name in names:
        checkpoint_mean = np.mean([[row["scores"][str(seed)][name] for row in rows] for seed in cfg["seeds"]], axis=0)
        np.testing.assert_allclose(checkpoint_mean, [row["scores"]["three_seed_mean"][name] for row in rows], atol=1e-15, rtol=0)
    labels = np.array([row["corpus"] for row in rows])
    for seed in [*map(str, cfg["seeds"]), "three_seed_mean"]:
        scores = {name: np.array([row["scores"][seed][name] for row in rows]) for name in names}
        for name, (a, b) in CONTRASTS.items():
            np.testing.assert_allclose(scores[a] - scores[b], [row["contrasts"][seed][name] for row in rows], atol=1e-15, rtol=0)
        np.testing.assert_allclose(scores["direct_four"] - scores["recursive_binary"], [row["degradation_direct_minus_recursive"][seed] for row in rows], atol=1e-15, rtol=0)
        for index, scope in enumerate(result["scope_order"]):
            mask = np.ones(606, dtype=bool) if scope == "pooled" else labels == scope
            expected = summarize_tree_scores(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + index, cfg["loss_thresholds"])
            assert expected == result["summaries"][seed][scope]


def test_full_population_reproduces_from_prior_metadata_and_original_histograms():
    cfg, _, _, published = load_protocol()
    prior = json.loads((ROOT / "results/safe_baselines_v1/DOCUMENT_METADATA.json").read_text())
    assert characterize_population(prior, cfg, published) == read("FULL_POPULATION.json")
    full = read("FULL_POPULATION.json")["pooled"]
    assert full["depth_histogram"] == {"1": 3631, "2": 349, "3": 20}
    assert full["documents_strictly_above_context"] == 81
    assert full["documents_with_recursive_reuse"] == 369
    assert [row["documents"] for row in full["length_bins"]] == [2707, 924, 288, 61, 15, 5]
    table = (OUTPUT / "full_population_table.tex").read_text()
    for row in full["length_bins"]:
        assert f" & {row['documents']:,} & {row['fraction'] * 100:.3f}" in table


def test_reused_direct_and_pooling_scores_match_the_exact_four_preceding_subset():
    prior = {row["id"]: row for row in json.loads((ROOT / "results/safe_baselines_v1/DOCUMENT_SCORES.json").read_text())}
    metadata = {row["id"]: row for row in read("DOCUMENT_METADATA.json")}
    for row in read("DOCUMENT_SCORES.json"):
        before = prior[row["id"]]
        assert row["corpus"] == before["corpus"]
        counts = metadata[row["id"]]["leaf_counts"]
        assert len(counts) == 4 and all(n > 0 for n in counts)
        assert metadata[row["id"]]["intermediate_counts"] == [sum(counts[:2]), sum(counts[2:])]
        for seed in (424242, 271828, 314159):
            for new, old in (("direct_four", f"learned_parent_seed{seed}"), ("flat_unweighted", "unweighted_mean"), ("flat_token_weighted", "token_weighted_mean")):
                assert abs(row["scores"][str(seed)][new] - before["cosines"][old]) < 1e-14
