"""Published calibration checks without checkpoint/cache/GPU requirements."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from recursive_geodesic_composition.reconstruction_calibration import METHODS, CONTRASTS, reconstruction_summary
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/reconstruction_calibration_v1"
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_reconstruction_calibration import score_matrices


def read(name):
    return json.loads((OUTPUT / name).read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_calibration_artifacts_sources_and_independent_audit():
    for name, expected in read("ARTIFACT_HASHES.json").items():
        path = OUTPUT / name
        if name == "vectors.pt" and not path.exists():
            continue
        assert sha(path) == expected, name
    result, audit, manifest = read("RESULTS.json"), read("AUDIT.json"), read("INFERENCE_MANIFEST.json")
    assert audit["passed"] and all(audit["checks"].values())
    assert (audit["documents"], audit["checkpoints"], audit["permutations"], audit["derangements"]) == (606, 3, 24, 9)
    assert audit["results_sha256"] == sha(OUTPUT / "RESULTS.json")
    assert not manifest["oracle_inputs"] and manifest["decoder_inputs"] == ["parent_vector"]
    assert manifest["new_training_updates"] == manifest["new_teacher_encoding_calls"] == manifest["new_merger_forward_calls"] == 0
    for section in ("source_sha256", "immutable_input_sha256"):
        assert result[section] == manifest[section]
        for path, expected in result[section].items():
            assert sha(ROOT / path) == expected, path
    assert max(audit["independent_replay_max_abs_differences"].values()) < result["configuration"]["independent_replay_max_absolute_tolerance"]


def test_all_assignment_scores_paired_contrasts_and_intervals_match_published_matrices():
    result, cross, docs = read("RESULTS.json"), read("CROSS_COSINES.json"), read("DOCUMENT_SCORES.json")
    cfg = result["configuration"]
    assert cross["groups"] == [row["id"] for row in docs]
    assert len(docs) == len(set(cross["groups"])) == 606
    scores = score_matrices({key: np.asarray(value) for key, value in cross["matrices"].items()}, cfg)
    labels = np.asarray(cross["labels"])
    for seed, methods in scores.items():
        for method in METHODS:
            for metric, values in methods[method].items():
                np.testing.assert_array_equal(values, [row["scores"][seed][method][metric] for row in docs])
            if "copy" in method:
                assert np.all(methods[method]["ordered_minus_shuffled"] == 0)
                assert np.all(methods[method]["ordered_minus_deranged"] == 0)
        for name, (a, b) in CONTRASTS.items():
            np.testing.assert_array_equal(methods[a]["ordered"] - methods[b]["ordered"], [row["contrasts"][seed][name] for row in docs])
        for i, scope in enumerate(result["scope_order"]):
            mask = np.ones(606, dtype=bool) if scope == "pooled" else labels == scope
            assert reconstruction_summary(methods, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + i) == result["summaries"][seed][scope]


def test_published_table_and_paper_report_copy_and_order_controls():
    from evaluate_reconstruction_calibration import LABELS
    result = read("RESULTS.json")
    s = result["summaries"]["three_seed_mean"]["pooled"]
    table = (OUTPUT / "reconstruction_calibration_table.tex").read_text()
    for method in METHODS:
        m = s["methods"][method]
        ordered, shuffled, gap = [m[key]["mean"] for key in ("ordered", "shuffled_all24", "ordered_minus_shuffled")]
        gain = ordered - s["methods"]["parent_copy"]["ordered"]["mean"]
        assert f"{LABELS[method]} & {ordered:.6f} & {gain:+.6f} & {shuffled:.6f} & {gap:+.8f}" in table


def test_historical_diagnostic_discrepancies_are_retained_not_suppressed():
    r = read("RESULTS.json")
    all_cells = [(seed, scope, method, cell) for seed, scopes in r["historical_reproduction_diagnostics"].items()
                 for scope, methods in scopes.items() for method, cell in methods.items()]
    failures = [(seed, scope, method) for seed, scope, method, cell in all_cells if not cell["within_fixed_diagnostic_tolerance"]]
    assert len(all_cells) == 30 and len(failures) == 6
    assert {scope for _, scope, _ in failures} == {"openwebtext"}
    assert all(cell["within_fixed_diagnostic_tolerance"] for _, scope, _, cell in all_cells if scope == "pooled")
    report = (OUTPUT / "REPORT.md").read_text()
    assert report.count("| False |") == 6
    assert r["no_matched_pooling_probe_replay"]


def test_geometric_scope_and_aggregation_are_correct():
    r, docs = read("RESULTS.json"), read("DOCUMENT_SCORES.json")
    keys = ("fraction_children_outside_cap", "loose_cap_cosine_ceiling")
    for key in keys:
        mean = np.mean([[row["geometry"][str(seed)][key] for row in docs] for seed in r["configuration"]["seeds"]], axis=0)
        np.testing.assert_array_equal(mean, [row["geometry"]["three_seed_mean"][key] for row in docs])
        assert np.mean(mean) == r["geometry"]["three_seed_mean"]["pooled"][key]["mean"]
    for method in ("strict_decoder", "frozen_probe"):
        scored = np.array([row["scores"]["three_seed_mean"][method]["ordered"] for row in docs])
        ceiling = np.array([row["geometry"]["three_seed_mean"]["loose_cap_cosine_ceiling"] for row in docs])
        assert np.all(scored <= ceiling + 1e-12)
