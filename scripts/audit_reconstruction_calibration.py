#!/usr/bin/env python3
"""Independent decoder replay and scalar/assignment calibration audit."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_reconstruction_calibration import CONFIG, inputs, load_protocol, load_inference, load_probe, predict, score_matrices
from safe_baselines_protocol import ROOT, atomic_json, configure_device, require_hash, sha256
from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.reconstruction_calibration import METHODS, reconstruction_summary


def audit(config_path, device_name):
    cfg, tree, baseline, _, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    root = ROOT / cfg["output_dir"]
    result = json.loads((root / "RESULTS.json").read_text())
    for file, field in (("INFERENCE_MANIFEST.json", "inference_manifest_sha256"), ("DOCUMENT_SCORES.json", "document_scores_sha256"), ("CROSS_COSINES.json", "cross_cosines_sha256")):
        require_hash(root / file, result[field])
    payload, manifest = load_inference(cfg)
    x, parents, groups, labels, _ = inputs(tree, baseline, published)
    assert len(groups) == len(set(groups)) == 606
    assert groups == payload["groups"] and labels == payload["labels"]
    assert torch.equal(x, payload["children"])
    assert all(torch.equal(value, payload["parents"][key]) for key, value in parents.items())
    saved = json.loads((root / "CROSS_COSINES.json").read_text())
    docs = json.loads((root / "DOCUMENT_SCORES.json").read_text())
    assert saved["groups"] == groups == [row["id"] for row in docs]
    matrices = {key: np.asarray(value) for key, value in saved["matrices"].items()}
    assignments = list(itertools.permutations(range(4)))
    for key, matrix in matrices.items():
        if key in payload["predictions"]:
            pred = payload["predictions"][key]
        else:
            tag = key.split("seed")[-1] if "seed" in key else key.removesuffix("_copy")
            pred = parents[tag][:, None, :].expand(-1, 4, -1)
        actual = F.cosine_similarity(pred.double()[:, :, None], x.double()[:, None, :], dim=-1).numpy()
        np.testing.assert_allclose(matrix, actual, atol=1e-14, rtol=0)
        # Exhaust assignments with independent loops, rather than evaluator indexing.
        perm = np.asarray([sum(actual[:, i, p[i]] for i in range(4)) / 4 for p in assignments]).T
        np.testing.assert_allclose(perm.mean(1), actual.mean((1, 2)), atol=1e-14, rtol=0)
        off = (actual.sum((1, 2)) - np.trace(actual, axis1=1, axis2=2)) / 12
        deranged = [j for j, p in enumerate(assignments) if all(i != p[i] for i in range(4))]
        np.testing.assert_allclose(perm[:, deranged].mean(1), off, atol=1e-14, rtol=0)
        seed = key.split("seed")[-1] if "seed" in key else str(cfg["seeds"][0])
        method = key.split("_seed")[0]
        for name, values in (("ordered", perm[:, 0]), ("shuffled_all24", perm.mean(1)),
                             ("shuffled_deranged9", perm[:, deranged].mean(1)), ("oracle_best_assignment", perm.max(1))):
            np.testing.assert_allclose(values, [row["scores"][seed][method][name] for row in docs], atol=1e-14, rtol=0)
    scores = score_matrices(matrices, cfg)
    labels_array = np.asarray(labels)
    for seed, methods in scores.items():
        for method in METHODS:
            for name, values in methods[method].items():
                np.testing.assert_allclose(values, [row["scores"][seed][method][name] for row in docs], atol=1e-14, rtol=0)
            if "copy" in method:
                assert np.max(np.abs(methods[method]["ordered_minus_shuffled"])) == 0
                assert np.max(np.abs(methods[method]["ordered_minus_deranged"])) == 0
        for i, scope in enumerate(result["scope_order"]):
            mask = np.ones(606, dtype=bool) if scope == "pooled" else labels_array == scope
            expected = reconstruction_summary(methods, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + i)
            assert expected == result["summaries"][seed][scope]
    # The unweighted normalized mean maximizes equal-child cosine among all
    # single unit vectors copied into every slot. This is NOT a decoder bound.
    optimum = F.normalize(x.double(), dim=-1).mean(1).norm(dim=-1).numpy()
    np.testing.assert_allclose(matrices["unweighted_copy"][:, 0].mean(1), optimum, atol=1e-12, rtol=0)
    replay, movements = {}, {}
    radius = np.arcsin(cfg["geometry"]["residual_norm_bound"])
    for entry, probe_entry in zip(manifest["checkpoints"], manifest["probes"]):
        seed = entry["seed"]
        model, checkpoint = build_model(ROOT / entry["path"], device)
        # Bypass the public decode wrapper; use the component directly and B=17.
        direct = predict(lambda p: model.strict_decoder(p)[0], parents[str(seed)], device, cfg["independent_replay_batch_size"])
        del model, checkpoint
        probe, _ = load_probe(probe_entry, device)
        probe_prediction = predict(probe, parents[str(seed)], device, cfg["independent_replay_batch_size"])
        del probe
        for method, replayed in (("strict_decoder", direct), ("frozen_probe", probe_prediction)):
            key = f"{method}_seed{seed}"
            difference = float((replayed - payload["predictions"][key]).abs().max())
            assert difference <= cfg["independent_replay_max_absolute_tolerance"]
            replay[key] = difference
            cosine = F.cosine_similarity(replayed.double(), parents[str(seed)].double()[:, None], dim=-1).clamp(-1, 1)
            angle = cosine.acos()
            assert float(angle.max()) <= radius + 1e-6
            movements[key] = {"maximum_radians": float(angle.max()), "mean_radians": float(angle.mean())}
        parent_cos = F.cosine_similarity(parents[str(seed)].double()[:, None], x.double(), dim=-1).clamp(-1, 1)
        outside = (parent_cos.acos() > radius).double().mean(1).numpy()
        ceiling = torch.cos((parent_cos.acos() - radius).clamp_min(0)).mean(1).numpy()
        np.testing.assert_allclose(outside, [row["geometry"][str(seed)]["fraction_children_outside_cap"] for row in docs], atol=0, rtol=0)
        np.testing.assert_allclose(ceiling, [row["geometry"][str(seed)]["loose_cap_cosine_ceiling"] for row in docs], atol=1e-14, rtol=0)
        for artifact in (entry, probe_entry):
            require_hash(ROOT / artifact["path"], artifact["sha256"])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    verdict = {"passed": True, "documents": 606, "checkpoints": 3, "permutations": 24, "derangements": 9,
               "checks": {"same_parent_and_child_tensors": True, "all_model_predictions_independently_replayed": True,
                          "parent_only_decoder_inputs": manifest["decoder_inputs"] == ["parent_vector"] and not manifest["oracle_inputs"],
                          "cross_cosines_recomputed_in_torch_double": True, "exact_permutation_algebra_verified": True,
                          "shared_copy_optimum_and_zero_order_advantage": True, "paired_intervals_recomputed": True,
                          "angular_bound_and_target_feasibility_checked": True, "checkpoint_probe_and_previous_results_unchanged": True},
               "independent_replay_max_abs_differences": replay, "decoder_movement": movements,
               "results_sha256": sha256(root / "RESULTS.json"), "audit_implementation_sha256": sha256(__file__)}
    assert all(verdict["checks"].values())
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
