#!/usr/bin/env python3
"""Independently replay source models; audit paired scores, classifiers and lineage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from evaluate_continuation_comparison import CONFIG, load_protocol, load_inference, inputs, artifacts, load_source_probe, verify_lineage, score_outputs
from safe_baselines_protocol import ROOT, atomic_json, configure_device, require_hash, sha256
from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.continuation_comparison import DocumentWeightedAUC, paired_auc, paired_summary


def audit(config_path, device_name):
    cfg, calibration, baseline, _, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    root = ROOT / cfg["output_dir"]
    result = json.loads((root / "RESULTS.json").read_text())
    for path, field in (("POINTS.json", "points_sha256"), ("PROBE_CROSS_COSINES.json", "probe_cross_cosines_sha256"),
                        ("SELECTED_CLASSIFIER_SCORES.json", "selected_classifier_scores_sha256"), ("INFERENCE_MANIFEST.json", "inference_manifest_sha256")):
        require_hash(root / path, result[field])
    for relative, digest in result["source_attack_sha256"].items():
        require_hash(root / relative, digest)
    for path, digest in result["reused_final_attack_sha256"].items():
        require_hash(ROOT / path, digest)
    payload, manifest = load_inference(cfg)
    safe, cal, rows, exact, _, cal_manifest = inputs(calibration, baseline, published)
    entries = artifacts(baseline)
    assert manifest["artifacts"] == entries
    assert payload["groups"] == safe["groups"].tolist() and len(set(payload["groups"])) == 4000
    assert payload["probe_groups"] == cal["groups"] and len(set(cal["groups"])) == 606
    assert payload["labels"] == safe["labels"].tolist() and payload["active_children"] == safe["active_children"].tolist()
    assert exact.tolist() == payload["exact_four_indices"]
    assert torch.equal(payload["children"], cal["children"])
    assert np.array_equal(payload["targets"].numpy(), safe["targets"])
    assert np.array_equal(payload["weighted_parents"].numpy(), safe["vectors"]["token_weighted_mean"])
    replay = {}
    with torch.inference_mode():
        for seed in map(str, cfg["seeds"]):
            assert np.array_equal(payload["final_parents"][seed].numpy(), safe["vectors"][f"learned_parent_seed{seed}"])
            assert torch.equal(payload["final_probes"][seed], cal["predictions"][f"frozen_probe_seed{seed}"])
            model, source = build_model(ROOT / entries[seed]["source"]["path"], device)
            final = torch.load(ROOT / entries[seed]["final"]["path"], weights_only=True, map_location="cpu")
            assert verify_lineage(source, final, entries[seed]["source"]) == manifest["lineage"][seed]
            del source, final
            roots = []
            # Different packing: original document order, mixed arities, B=17.
            # Do not reuse the evaluator's arity-grouped padded_batches helper.
            size = cfg["independent_replay_batch_size"]
            for start in range(0, len(rows), size):
                part = rows[start:start + size]
                x = torch.zeros(len(part), 4, 768, device=device)
                n = torch.zeros(len(part), 4, device=device)
                for j, row in enumerate(part):
                    k = len(row["lengths"])
                    x[j, :k] = row["leaves"].to(device)
                    n[j, :k] = row["lengths"].to(device)
                roots.append(model.merge(x, n).cpu())
            predicted_roots = torch.cat(roots)
            del model
            probe, metadata = load_source_probe(entries[seed]["source_probe"], cal_manifest["probe_training_provenance"][seed], device)
            assert metadata == manifest["source_probe_metadata"][seed]
            # Replay probes from the independently recomputed parents, not saved roots.
            predicted_children = torch.cat([probe(part.to(device)).cpu() for part in predicted_roots[exact].split(size)])
            del probe
            replay[seed] = {"source_parent": float((predicted_roots - payload["source_parents"][seed]).abs().max()),
                            "source_probe": float((predicted_children - payload["source_probes"][seed]).abs().max())}
            assert max(replay[seed].values()) <= cfg["independent_replay_max_absolute_tolerance"]
            assert torch.isfinite(predicted_roots).all() and torch.isfinite(predicted_children).all()
            assert torch.allclose(predicted_roots.norm(dim=-1), torch.ones(4000), rtol=0, atol=2e-5)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    points = json.loads((root / "POINTS.json").read_text())
    cross = json.loads((root / "PROBE_CROSS_COSINES.json").read_text())
    teacher, retention, matrices = score_outputs(payload, cfg)
    labels = np.asarray(payload["labels"])
    for seed in map(str, cfg["seeds"]):
        for stage in ("source", "final"):
            value = F.cosine_similarity(payload[f"{stage}_parents"][seed].double(), payload["targets"].double(), dim=-1).numpy()
            np.testing.assert_allclose(value, teacher[seed][stage], atol=1e-14, rtol=0)
            for kind in ("probe", "copy"):
                prediction = payload[f"{stage}_probes"][seed] if kind == "probe" else payload[f"{stage}_parents"][seed][exact, None].expand(-1, 4, -1)
                matrix = F.cosine_similarity(prediction.double()[:, :, None], payload["children"].double()[:, None], dim=-1).numpy()
                key = f"{stage}_{kind}_seed{seed}"
                np.testing.assert_allclose(matrix, cross["matrices"][key], atol=1e-14, rtol=0)
                np.testing.assert_array_equal(matrices[key], cross["matrices"][key])
                np.testing.assert_allclose(np.trace(matrix, axis1=1, axis2=2) / 4, retention[seed][f"{stage}_{kind}"], atol=1e-14, rtol=0)
                if kind == "probe":
                    np.testing.assert_allclose(np.trace(matrix, axis1=1, axis2=2) / 4 - matrix.mean((1, 2)), retention[seed][f"{stage}_order_gap"], atol=1e-12, rtol=0)
    for section, values in (("teacher", teacher), ("reconstruction", retention)):
        for seed, scores in values.items():
            for key, value in scores.items():
                np.testing.assert_array_equal(value, points[section][seed][key])
            for i, scope in enumerate(result["fidelity_scope_order"] if section == "teacher" else result["scope_order"]):
                mask = np.ones(4000, dtype=bool) if scope == "pooled" else np.asarray(payload["active_children"]) == int(scope[-1]) if scope.startswith("arity_") else labels == scope
                if section == "reconstruction":
                    mask = mask[exact]
                assert paired_summary(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["paired_score_seed"] + i) == result[section][seed][scope]
    selected = json.loads((root / "SELECTED_CLASSIFIER_SCORES.json").read_text())
    maximum_auc_error = 0.
    for i, scope in enumerate(result["scope_order"]):
        pair = selected[scope]
        recalculated = paired_auc(pair["source"], pair["final"], cfg["bootstrap"]["reps"], cfg["bootstrap"]["paired_auc_seed"] + i)
        assert recalculated == {k: v for k, v in result["source_detection"][scope].items() if k != "selected_families"}
        for j, seed in enumerate(cfg["seeds"]):
            source = json.loads((root / f"attacks/source_seed{seed}__{scope}.json").read_text())
            final = json.loads((ROOT / baseline["output_dir"] / f"attacks/learned_parent_seed{seed}__{scope}.json").read_text())
            assert source["classifier_train_groups"] == final["classifier_train_groups"]
            assert source["classifier_holdout_groups"] == final["classifier_holdout_groups"]
            assert set(source["classifier_train_groups"]).isdisjoint(source["classifier_holdout_groups"])
            for stage, original in (("source", source), ("final", final)):
                family = original["metrics"]["strongest_attack"]
                assert pair[stage][j] == original["bundles"][family]
                assert family == result["source_detection"][scope]["selected_families"][str(seed)][stage]
                bundle = pair[stage][j]
                evaluator = DocumentWeightedAUC(bundle)
                n = len(evaluator.documents)
                raw = roc_auc_score(bundle["y"], bundle["scores"])
                assert abs(max(raw, 1 - raw) - original["metrics"]["strongest_orientation_safe_auc"]) < 1e-14
                rng = np.random.default_rng(cfg["bootstrap"]["paired_auc_seed"] + i)
                inverse = np.searchsorted(evaluator.documents, bundle["groups"])
                for _ in range(cfg["independent_auc_bootstrap_checks_per_scope"]):
                    counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
                    independent = roc_auc_score(bundle["y"], bundle["scores"], sample_weight=counts[inverse])
                    error = abs(independent - evaluator.raw(counts))
                    maximum_auc_error = max(maximum_auc_error, error)
                    assert error < 1e-14
    for path, digest in manifest["immutable_input_sha256"].items():
        require_hash(ROOT / path, digest)
    artifacts(baseline)
    verdict = {"passed": True, "safe_documents": 4000, "probe_documents": 606, "checkpoints": 3,
        "checks": {"source_final_lineage_and_probe_metadata_matched": True, "all_previous_inputs_outputs_unchanged": True,
                   "source_parents_and_probes_independently_replayed": True, "double_precision_cosines_and_paired_intervals_verified": True,
                   "same_document_classifier_splits_and_frozen_family_choices": True, "all_auc_intervals_recomputed": True,
                   "weighted_auc_independently_checked_with_sklearn": True},
        "independent_replay_max_abs_differences": replay, "independent_auc_resamples_per_bundle": cfg["independent_auc_bootstrap_checks_per_scope"],
        "independent_auc_max_absolute_error": maximum_auc_error, "results_sha256": sha256(root / "RESULTS.json"),
        "audit_implementation_sha256": sha256(__file__)}
    atomic_json(root / "AUDIT.json", verdict)
    if (root / "ARTIFACT_HASHES.json").exists():
        hashes = json.loads((root / "ARTIFACT_HASHES.json").read_text())
        hashes["AUDIT.json"] = sha256(root / "AUDIT.json")
        atomic_json(root / "ARTIFACT_HASHES.json", hashes)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    audit(args.config, args.device)
