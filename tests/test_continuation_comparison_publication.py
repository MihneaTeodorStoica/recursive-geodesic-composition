"""Published continuation checks; no caches, model weights, or GPU needed."""
import hashlib
import json
from pathlib import Path
import numpy as np
from recursive_geodesic_composition.continuation_comparison import paired_summary, paired_auc
from recursive_geodesic_composition.reconstruction_calibration import assignment_scores

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/continuation_comparison_v1"


def read(name):
    return json.loads((OUTPUT / name).read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_continuation_integrity_and_independent_replay():
    for name, expected in read("ARTIFACT_HASHES.json").items():
        path = OUTPUT / name
        if name == "vectors.pt" and not path.exists():
            continue
        assert digest(path) == expected, name
    r, m, a = read("RESULTS.json"), read("INFERENCE_MANIFEST.json"), read("AUDIT.json")
    assert a["passed"] and all(a["checks"].values())
    assert (a["safe_documents"], a["probe_documents"], a["checkpoints"]) == (4000, 606, 3)
    assert a["results_sha256"] == digest(OUTPUT / "RESULTS.json")
    for section in ("source_sha256", "immutable_input_sha256"):
        assert r[section] == m[section]
        for file, expected in r[section].items():
            assert digest(ROOT / file) == expected
    assert all(max(v.values()) <= r["configuration"]["independent_replay_max_absolute_tolerance"] for v in a["independent_replay_max_abs_differences"].values())
    assert a["independent_auc_max_absolute_error"] < 1e-14
    assert m["new_model_updates"] == m["new_teacher_encoding_calls"] == 0
    assert m["final_predictions_reused_without_change"] and r["not_an_objective_isolating_ablation"]
    presentation = read("PRESENTATION.json")
    assert presentation["layout_only_after_results"] and presentation["all_primary_outputs_unchanged"]
    assert presentation["results_sha256"] == a["results_sha256"]
    assert presentation["implementation_sha256"] == digest(ROOT / "scripts/publish_continuation_comparison.py")
    assert presentation["frozen_renderer_sha256"] == r["source_sha256"]["scripts/render_continuation_comparison.py"]


def test_continuation_lineage_and_probe_train_metadata_are_matched():
    m = read("INFERENCE_MANIFEST.json")
    original = json.loads((ROOT / "results/reconstruction_calibration_v1/INFERENCE_MANIFEST.json").read_text())
    for seed, lineage in m["lineage"].items():
        source = m["artifacts"][seed]["source"]
        stage = lineage["staged_training"]
        assert stage["initialization_checkpoint_sha256"] == source["sha256"]
        assert stage["initialization_checkpoint"] == source["protocol_path"]
        assert lineage["merger_tensors"] == lineage["changed_merger_tensors"] == 113
        assert stage["initialization_ignored_decoder_key_count"] == 19
        assert stage["freeze_encoder_steps"] == 300 and stage["encoder_learning_rate_scale"] == .02
        for field in ("selection_audit", "schedule_sha256", "seed", "fit_role"):
            assert m["source_probe_metadata"][seed][field] == original["probe_training_provenance"][seed][field]


def test_scalar_population_pairing_and_paired_bootstrap():
    r, points = read("RESULTS.json"), read("POINTS.json")
    cfg = r["configuration"]
    assert len(points["groups"]) == len(set(points["groups"])) == 4000
    exact = np.flatnonzero(np.asarray(points["active_children"]) == 4)
    assert len(exact) == 606 and exact.tolist() == points["exact_four_indices"]
    assert np.asarray(points["groups"])[exact].tolist() == points["probe_groups"]
    labels = np.asarray(points["labels"])
    for section in ("teacher", "reconstruction"):
        for key, values in points[section]["three_seed_mean"].items():
            np.testing.assert_array_equal(np.mean([points[section][str(seed)][key] for seed in cfg["seeds"]], axis=0), values)
        for seed, scores in points[section].items():
            for i, scope in enumerate(r["fidelity_scope_order"] if section == "teacher" else r["scope_order"]):
                mask = np.ones(4000, bool) if scope == "pooled" else np.asarray(points["active_children"]) == int(scope[-1]) if scope.startswith("arity_") else labels == scope
                if section == "reconstruction":
                    mask = mask[exact]
                assert paired_summary(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["paired_score_seed"] + i) == r[section][seed][scope]
            if seed != "three_seed_mean":
                if section == "teacher":
                    np.testing.assert_array_equal(np.array(scores["final"]) - scores["source"], scores["delta"])
                else:
                    for metric in ("probe", "copy", "gain", "order_gap"):
                        np.testing.assert_array_equal(np.array(scores[f"final_{metric}"]) - scores[f"source_{metric}"], scores[f"{metric}_delta"])


def test_final_reconstruction_matrices_unchanged_and_source_copy_calibration():
    r, cross, points = read("RESULTS.json"), read("PROBE_CROSS_COSINES.json"), read("POINTS.json")
    previous = json.loads((ROOT / "results/reconstruction_calibration_v1/CROSS_COSINES.json").read_text())
    assert cross["groups"] == previous["groups"] == points["probe_groups"]
    for seed in map(str, r["configuration"]["seeds"]):
        for kind, old in (("probe", "frozen_probe"), ("copy", "parent_copy")):
            np.testing.assert_array_equal(cross["matrices"][f"final_{kind}_seed{seed}"], previous["matrices"][f"{old}_seed{seed}"])
        for stage in ("source", "final"):
            probe = assignment_scores(cross["matrices"][f"{stage}_probe_seed{seed}"])
            copy = assignment_scores(cross["matrices"][f"{stage}_copy_seed{seed}"])
            p = points["reconstruction"][seed]
            np.testing.assert_array_equal(p[f"{stage}_probe"], probe["ordered"])
            np.testing.assert_array_equal(p[f"{stage}_copy"], copy["ordered"])
            np.testing.assert_array_equal(p[f"{stage}_gain"], probe["ordered"] - copy["ordered"])
            np.testing.assert_array_equal(p[f"{stage}_order_gap"], probe["ordered_minus_shuffled"])
            assert not np.any(copy["ordered_minus_shuffled"])


def test_all_auc_intervals_scores_split_identities_and_family_choices():
    r, selected = read("RESULTS.json"), read("SELECTED_CLASSIFIER_SCORES.json")
    cfg = r["configuration"]
    assert r["source_classifier_fits"] == 30 and r["final_classifier_fits"] == 0
    assert len(r["source_attack_sha256"]) == len(r["reused_final_attack_sha256"]) == 15
    for i, scope in enumerate(r["scope_order"]):
        pair = selected[scope]
        actual = paired_auc(pair["source"], pair["final"], cfg["bootstrap"]["reps"], cfg["bootstrap"]["paired_auc_seed"] + i)
        assert actual == {k: v for k, v in r["source_detection"][scope].items() if k != "selected_families"}
        for j, seed in enumerate(cfg["seeds"]):
            source = read(f"attacks/source_seed{seed}__{scope}.json")
            path = f"results/safe_baselines_v1/attacks/learned_parent_seed{seed}__{scope}.json"
            final = json.loads((ROOT / path).read_text())
            assert digest(ROOT / path) == r["reused_final_attack_sha256"][path]
            for field in ("classifier_train_groups", "classifier_holdout_groups", "attack_recipe_sha256"):
                assert source[field] == final[field]
            assert set(source["classifier_train_groups"]).isdisjoint(source["classifier_holdout_groups"])
            for stage, entry in (("source", source), ("final", final)):
                family = entry["metrics"]["strongest_attack"]
                assert r["source_detection"][scope]["selected_families"][str(seed)][stage] == family
                assert pair[stage][j] == entry["bundles"][family]
                np.testing.assert_allclose(r["source_detection"][scope]["per_checkpoint"][j][stage]["mean"],
                                           entry["metrics"]["strongest_orientation_safe_auc"], atol=1e-14, rtol=0)


def test_paper_table_and_primary_contrasts_are_linked_to_outputs():
    r = read("RESULTS.json")
    table = (OUTPUT / "continuation_comparison_table.tex").read_text()
    paper = (ROOT / "paper/main.tex").read_text()
    assert "../results/continuation_comparison_v1/continuation_comparison_table.tex" in paper
    assert "../results/continuation_comparison_v1/continuation_ecdf.png" in paper
    teacher = r["teacher"]["three_seed_mean"]["pooled"]["delta"]
    assert f"{teacher['mean'] * 1e6:+.3f}" in table
    assert r"Change [95\% CI] $(\times10^{-6})$" in table
    assert f"{teacher['mean'] * 1e7:+.4f}" in paper
    assert f"[{teacher['ci95'][0] * 1e7:.4f},{teacher['ci95'][1] * 1e7:.4f}]" in paper
    for delta in (r["source_detection"]["pooled"]["three_seed_mean"]["final_minus_source"],
                  r["reconstruction"]["three_seed_mean"]["pooled"]["gain_delta"]):
        assert f"{delta['mean'] * 1e6:+.3f}" in table
        assert f"[{delta['ci95'][0] * 1e6:+.3f}, {delta['ci95'][1] * 1e6:+.3f}]" in table
        assert f"{delta['mean']:+.6f}" in paper
        assert f"[{delta['ci95'][0]:.6f},{delta['ci95'][1]:.6f}]" in paper
    assert "not a decoder-objective" in paper


def test_supplementary_numerical_check_is_explicitly_post_result_and_non_substitutive():
    protocol, check = read("NUMERICAL_CHECK_PROTOCOL.json"), read("NUMERICAL_CHECK.json")
    assert protocol["role"] == "supplementary_post_result_numerical_validation"
    assert protocol["retain_all_shifts_without_result_substitution"] and protocol["not_a_new_primary_analysis"]
    assert check["results_sha256"] == digest(OUTPUT / "RESULTS.json")
    assert check["implementation_sha256"] == digest(ROOT / "scripts/check_continuation_numerics.py")
    assert all(check[key] == value for key, value in protocol.items())
    shifts = [check["outcomes"][str(seed)]["delta"]["mean_shift"] for seed in check["seeds"]]
    np.testing.assert_allclose(np.mean(shifts), check["three_seed_mean_delta_shift"], atol=1e-20, rtol=0)
    for checkpoint in check["outcomes"].values():
        for stage in ("source", "final"):
            assert checkpoint[stage]["maximum_absolute_vector_error"] <= read("RESULTS.json")["configuration"]["independent_replay_max_absolute_tolerance"]
