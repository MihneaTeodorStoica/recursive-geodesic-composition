#!/usr/bin/env python3
"""Audit recursive-reuse scores and replay the tree with independent packing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_recursive_reuse import (
    CONFIG, characterize_population, load_inference, load_protocol, select_exact_four,
)
from safe_baselines_protocol import (
    ROOT, atomic_json, checkpoint_entries, configure_device, load_safe_cache,
    require_hash, sha256, stable_hash,
)
from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.recursive_reuse import CONTRASTS, summarize_tree_scores


def independent_binary_tree(model, children, lengths):
    """Separate left/right forward calls: deliberately NOT the 2B packed helper."""
    left = model.merge(F.pad(children[:, :2], (0, 0, 0, 2)), F.pad(lengths[:, :2], (0, 2)))
    right = model.merge(F.pad(children[:, 2:], (0, 0, 0, 2)), F.pad(lengths[:, 2:], (0, 2)))
    parents = torch.stack((left, right), dim=1)
    masses = torch.stack((lengths[:, :2].sum(1), lengths[:, 2:].sum(1)), dim=1)
    root = model.merge(F.pad(parents, (0, 0, 0, 2)), F.pad(masses, (0, 2)))
    return root, parents, masses


def audit(config_path, device_name):
    cfg, baseline, official, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    root = ROOT / cfg["output_dir"]
    result = json.loads((root / "RESULTS.json").read_text())
    require_hash(root / "INFERENCE_MANIFEST.json", result["inference_manifest_sha256"])
    require_hash(root / "DOCUMENT_SCORES.json", result["document_scores_sha256"])
    payload, manifest = load_inference(cfg)
    rows, _ = load_safe_cache(baseline, published)
    selected, _ = select_exact_four(rows, cfg, published)
    assert payload["groups"] == [row["id"] for row in selected]
    assert stable_hash(payload["groups"]) == result["population_order_sha256"]
    assert len(payload["groups"]) == len(set(payload["groups"])) == 606
    assert torch.equal(payload["targets"], torch.stack([row["teacher_target"] for row in selected]))
    assert torch.equal(payload["counts"], torch.stack([row["lengths"] for row in selected]))
    assert torch.equal(payload["intermediate_counts"], payload["counts"].reshape(606, 2, 2).sum(2))
    assert torch.equal(payload["intermediate_counts"].sum(1), payload["counts"].sum(1))
    actual_population = characterize_population(rows, cfg, published)
    assert actual_population == json.loads((root / "FULL_POPULATION.json").read_text())
    documents = json.loads((root / "DOCUMENT_SCORES.json").read_text())
    assert [row["id"] for row in documents] == payload["groups"]
    # Score independently with PyTorch double precision, not NumPy's evaluator.
    by_seed = {}
    for seed in cfg["seeds"]:
        scores = {}
        for method in cfg["methods"]:
            key = f"{method}_seed{seed}" if method in ("direct_four", "recursive_binary") else method
            scores[method] = F.cosine_similarity(payload["vectors"][key].double(), payload["targets"].double(), dim=-1).numpy()
        scores["direct_recursive_output_cosine"] = F.cosine_similarity(
            payload["vectors"][f"direct_four_seed{seed}"].double(),
            payload["vectors"][f"recursive_binary_seed{seed}"].double(), dim=-1).numpy()
        by_seed[str(seed)] = scores
    by_seed["three_seed_mean"] = {name: np.mean([by_seed[str(seed)][name] for seed in cfg["seeds"]], axis=0)
                                   for name in by_seed[str(cfg["seeds"][0])]}
    labels = np.asarray(payload["labels"])
    for seed, scores in by_seed.items():
        for name, values in scores.items():
            np.testing.assert_allclose(values, [row["scores"][seed][name] for row in documents], atol=1e-14, rtol=0)
        for name, (a, b) in CONTRASTS.items():
            np.testing.assert_allclose(scores[a] - scores[b], [row["contrasts"][seed][name] for row in documents], atol=1e-14, rtol=0)
        for index, scope in enumerate(result["scope_order"]):
            mask = np.ones(len(labels), dtype=bool) if scope == "pooled" else labels == scope
            expected = summarize_tree_scores(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + index, cfg["loss_thresholds"])
            actual = result["summaries"][seed][scope]
            for section in ("metrics", "contrasts"):
                for name in expected[section]:
                    for field in ("mean", "ci95"):
                        np.testing.assert_allclose(expected[section][name][field], actual[section][name][field], atol=1e-14, rtol=0)
            for field in ("fraction_degraded", "fraction_improved", "fraction_tied", "fraction_loss_above"):
                assert expected["degradation_direct_minus_recursive"][field] == actual["degradation_direct_minus_recursive"][field]

    replay = {}
    with torch.inference_mode():
        for entry in checkpoint_entries(baseline):
            seed = entry["seed"]
            model, checkpoint = build_model(ROOT / entry["path"], device)
            maxima = {"direct": 0., "recursive": 0., "intermediate": 0.}
            # A different, deliberately awkward document batch size tests both
            # last-batch handling and the packed implementation's row alignment.
            for start in range(0, len(selected), 17):
                part = selected[start:start + 17]
                x = torch.stack([row["leaves"] for row in part]).float().to(device)
                n = torch.stack([row["lengths"] for row in part]).float().to(device)
                direct = model.merge(x, n).cpu()
                recursive, intermediate, masses = independent_binary_tree(model, x, n)
                end = start + len(part)
                assert torch.equal(masses.cpu(), payload["intermediate_counts"][start:end])
                for name, actual, expected in (
                    ("direct", direct, payload["vectors"][f"direct_four_seed{seed}"][start:end]),
                    ("recursive", recursive.cpu(), payload["vectors"][f"recursive_binary_seed{seed}"][start:end]),
                    ("intermediate", intermediate.cpu(), payload["intermediate_vectors"][str(seed)][start:end]),
                ):
                    difference = float((actual - expected).abs().max())
                    maxima[name] = max(maxima[name], difference)
                    if difference > cfg["independent_tree_max_absolute_tolerance"]:
                        raise ValueError(f"independent {name} replay failed for checkpoint {seed}: {difference}")
            replay[str(seed)] = maxima
            require_hash(ROOT / entry["path"], entry["sha256"])
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for path, digest in result["immutable_input_sha256"].items():
        require_hash(ROOT / path, digest)
    verdict = {
        "passed": True, "documents": 606, "checkpoints": 3,
        "checks": {"exact_four_population_and_target_identity": True,
                   "all_nodes_preserve_order_and_exact_token_counts": True,
                   "independent_tree_packing_replayed_for_every_document_and_checkpoint": True,
                   "independent_double_precision_scores_recomputed": True,
                   "paired_contrasts_and_intervals_recomputed": True,
                   "full_population_histograms_match_all_original_seeds": True,
                   "original_and_improvement_one_results_unchanged": True,
                   "checkpoint_hashes_unchanged": True},
        "independent_replay_max_abs_differences": replay,
        "independent_replay_batch_size": 17,
        "results_sha256": sha256(root / "RESULTS.json"), "audit_implementation_sha256": sha256(__file__),
    }
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
