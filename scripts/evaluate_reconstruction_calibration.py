#!/usr/bin/env python3
"""Additive parent-copy and target-slot permutation calibration, without fitting."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluate_recursive_reuse import load_protocol as load_tree_protocol, load_inference as load_tree_inference, select_exact_four
from safe_baselines_protocol import ROOT, atomic_json, atomic_tensor, checkpoint_entries, configure_device, environment, load_safe_cache, require_hash, sha256, stable_hash
from recursive_geodesic_composition.evaluate_kary_validation import build_model, ParentOnlyRetentionProbe
from recursive_geodesic_composition.reconstruction_calibration import METHODS, CONTRASTS, assignment_scores, cross_cosines, geometry_calibration, reconstruction_summary
from recursive_geodesic_composition.safe_baselines import validate_vectors, summarize_scores

CONFIG = ROOT / "configs/reconstruction_calibration_v1.yaml"
LABELS = {"parent_copy": "Learned parent copied", "strict_decoder": "Built-in strict decoder",
          "frozen_probe": "Released TRAIN-fitted probe", "weighted_copy": "Weighted pooling copied",
          "unweighted_copy": "Unweighted pooling copied"}


def load_protocol(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["study"] != "reconstruction_calibration_v1" or cfg["output_dir"] != "results/reconstruction_calibration_v1":
        raise ValueError("wrong additive namespace")
    if cfg["methods"] != list(METHODS) or cfg["seeds"] != [424242, 271828, 314159] or cfg["expected_documents"] != 606:
        raise ValueError("methods, checkpoints or population changed")
    if any(cfg[k] != 0 for k in ("merger_updates", "decoder_updates", "probe_updates", "teacher_encoding_calls", "classifier_fits")):
        raise ValueError("inference-only calibration; no fitting or new encoding")
    require_hash(ROOT / cfg["tree_config"], cfg["tree_config_sha256"])
    for file, digest in cfg["immutable_inputs"].items():
        require_hash(ROOT / file, digest)
    cfg["config_sha256"] = sha256(path)
    tree, baseline, official, published = load_tree_protocol(ROOT / cfg["tree_config"])
    return cfg, tree, baseline, official, published


def source_hashes():
    files = ["configs/reconstruction_calibration_v1.yaml", "scripts/evaluate_reconstruction_calibration.py",
             "scripts/render_reconstruction_calibration.py", "scripts/audit_reconstruction_calibration.py",
             "src/recursive_geodesic_composition/reconstruction_calibration.py",
             "src/recursive_geodesic_composition/architecture.py",
             "src/recursive_geodesic_composition/evaluate_kary_validation.py",
             "src/recursive_geodesic_composition/safe_baselines.py", "scripts/safe_baselines_protocol.py",
             "scripts/evaluate_recursive_reuse.py"]
    return {file: sha256(ROOT / file) for file in files}


def probe_entries(baseline):
    release = json.loads((ROOT / baseline["release_manifest"]).read_text())
    found = {row["seed"]: row for row in release["artifacts"] if row["role"] == "candidate_parent_only_fitted_probe"}
    result = []
    for seed in baseline["seeds"]:
        entry = dict(found[seed])
        entry["path"] = str(Path(baseline["artifact_dir"]) / entry["asset_filename"])
        path = ROOT / entry["path"]
        require_hash(path, entry["sha256"])
        if path.stat().st_size != entry["byte_size"]:
            raise ValueError("probe size mismatch")
        result.append(entry)
    return result


def load_probe(entry, device):
    require_hash(ROOT / entry["path"], entry["sha256"])
    checkpoint = torch.load(ROOT / entry["path"], map_location="cpu", weights_only=True)
    if (checkpoint.get("seed"), checkpoint.get("arm"), checkpoint.get("fit_role")) != (entry["seed"], "candidate", "fresh_v6_train"):
        raise ValueError("probe is not the released candidate TRAIN fit")
    old = json.loads((ROOT / "results/v1_k4_matched_pooling/PROBE_MANIFEST.json").read_text())
    if checkpoint["schedule_sha256"] != old["schedule_sha256"]:
        raise ValueError("TRAIN probe schedule identity changed")
    for corpus, selection in old["selection"].items():
        if checkpoint["selection_audit"][corpus]["fit_sha256"] != selection["group_order_sha256"]:
            raise ValueError("TRAIN probe fit population identity changed")
    probe = ParentOnlyRetentionProbe(768, 4, 512).to(device)
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    return probe.eval(), {key: value for key, value in checkpoint.items() if key != "state_dict"}


def predict(call, parents, device, batch_size):
    # The callable gets ONE tensor argument. No targets, IDs, counts or masks.
    with torch.inference_mode():
        return torch.cat([call(part.to(device)).float().cpu() for part in parents.split(batch_size)])


def inputs(tree, baseline, published):
    previous, manifest = load_tree_inference(tree)
    rows, _ = load_safe_cache(baseline, published)
    require_hash(ROOT / baseline["cache_dir"] / "CACHE_MANIFEST.json", manifest["cache_manifest_sha256"])
    selected, _ = select_exact_four(rows, tree, published)
    if previous["groups"] != [row["id"] for row in selected]:
        raise ValueError("parent/child document alignment changed")
    children = torch.stack([row["leaves"] for row in selected]).float()
    parents = {str(seed): previous["vectors"][f"direct_four_seed{seed}"] for seed in tree["seeds"]}
    parents |= {"weighted": previous["vectors"]["flat_token_weighted"], "unweighted": previous["vectors"]["flat_unweighted"]}
    return children, parents, previous["groups"], previous["labels"], manifest


def infer(config_path, device_name):
    cfg, tree, baseline, official, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    root = ROOT / cfg["output_dir"]
    if root.exists():
        raise FileExistsError("create-only output; use analyse/report/audit for completed inference")
    children, parents, groups, labels, prior_manifest = inputs(tree, baseline, published)
    checkpoints, probes = checkpoint_entries(baseline), probe_entries(baseline)
    immutable = prior_manifest["immutable_input_sha256"] | cfg["immutable_inputs"] | {cfg["tree_config"]: cfg["tree_config_sha256"]}
    freeze = {"configuration": cfg, "source_sha256": source_hashes(), "immutable_input_sha256": immutable,
              "environment": environment(device), "checkpoints": checkpoints, "probes": probes,
              "cache_manifest_sha256": prior_manifest["cache_manifest_sha256"], "started_unix": time.time()}
    root.mkdir(parents=True)
    atomic_json(root / "ANALYSIS_PROTOCOL.json", freeze)
    predictions, provenance, health = {}, {}, {}
    for entry, probe_entry in zip(checkpoints, probes):
        seed = entry["seed"]
        model, checkpoint = build_model(ROOT / entry["path"], device)
        if model.strict_decoder is None or model.arity != 4 or checkpoint["seed"] != seed:
            raise ValueError("wrong built-in decoder")
        predictions[f"strict_decoder_seed{seed}"] = predict(model.decode, parents[str(seed)], device, cfg["inference"]["batch_size"])
        del model, checkpoint
        probe, provenance[str(seed)] = load_probe(probe_entry, device)
        predictions[f"frozen_probe_seed{seed}"] = predict(probe, parents[str(seed)], device, cfg["inference"]["batch_size"])
        del probe
        for artifact in (entry, probe_entry):
            require_hash(ROOT / artifact["path"], artifact["sha256"])
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Completed decoder and frozen probe for {seed}", flush=True)
    for name, tensor in predictions.items():
        if tensor.shape != children.shape:
            raise ValueError("predictions do not have four aligned child slots")
        validate_vectors(tensor.reshape(-1, 768).numpy(), name, unit=True)
        health[name] = {"finite_unit_vectors": True, "shape": list(tensor.shape)}
    atomic_tensor(root / "vectors.pt", {"children": children, "parents": parents, "predictions": predictions, "groups": groups, "labels": labels})
    for path, digest in immutable.items():
        require_hash(ROOT / path, digest)
    atomic_json(root / "INFERENCE_MANIFEST.json", freeze | {
        "documents": len(groups), "population_order_sha256": stable_hash(groups), "vectors_sha256": sha256(root / "vectors.pt"),
        "health": health, "probe_training_provenance": provenance,
        "decoder_inputs": ["parent_vector"], "oracle_inputs": [], "new_training_updates": 0,
        "new_teacher_encoding_calls": 0, "new_merger_forward_calls": 0, "completed_unix": time.time()})


def load_inference(cfg):
    root = ROOT / cfg["output_dir"]
    manifest = json.loads((root / "INFERENCE_MANIFEST.json").read_text())
    if manifest["configuration"]["config_sha256"] != cfg["config_sha256"]:
        raise ValueError("configuration changed after inference")
    for section in ("source_sha256", "immutable_input_sha256"):
        for file, digest in manifest[section].items():
            require_hash(ROOT / file, digest)
    require_hash(root / "vectors.pt", manifest["vectors_sha256"])
    payload = torch.load(root / "vectors.pt", map_location="cpu", weights_only=True)
    if stable_hash(payload["groups"]) != manifest["population_order_sha256"]:
        raise ValueError("document order changed")
    return payload, manifest


def build_cross_matrices(payload):
    children = payload["children"].numpy()
    matrices = {key: cross_cosines(value.numpy(), children) for key, value in payload["predictions"].items()}
    for key, parent in payload["parents"].items():
        name = f"parent_copy_seed{key}" if key not in ("weighted", "unweighted") else f"{key}_copy"
        matrices[name] = cross_cosines(parent[:, None, :].expand(-1, 4, -1).numpy(), children)
    return matrices


def score_matrices(matrices, cfg):
    raw = {key: assignment_scores(value, cfg["numerical_zero_tolerance"]) for key, value in matrices.items()}
    by_seed = {str(seed): {method: raw[method] if method in ("weighted_copy", "unweighted_copy") else raw[f"{method}_seed{seed}"]
                          for method in METHODS} for seed in cfg["seeds"]}
    by_seed["three_seed_mean"] = {method: {metric: np.mean([by_seed[str(seed)][method][metric] for seed in cfg["seeds"]], axis=0)
                                           for metric in by_seed[str(cfg["seeds"][0])][method]} for method in METHODS}
    return by_seed


def analyse(config_path):
    cfg, _, _, _, published = load_protocol(config_path)
    root = ROOT / cfg["output_dir"]
    if (root / "RESULTS.json").exists():
        raise FileExistsError("results already exist")
    payload, manifest = load_inference(cfg)
    matrices = build_cross_matrices(payload)
    by_seed = score_matrices(matrices, cfg)
    scopes = {"pooled": np.ones(606, dtype=bool)} | {corpus: np.asarray(payload["labels"]) == corpus for corpus in published["corpus_order"]}
    summaries = {seed: {scope: reconstruction_summary(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + i)
                        for i, (scope, mask) in enumerate(scopes.items())} for seed, scores in by_seed.items()}
    geometry = {str(seed): geometry_calibration(matrices[f"parent_copy_seed{seed}"][:, 0], cfg["geometry"]["residual_norm_bound"])
                for seed in cfg["seeds"]}
    geometry["three_seed_mean"] = {key: np.mean([geometry[str(seed)][key] for seed in cfg["seeds"]], axis=0) for key in geometry[str(cfg["seeds"][0])]}
    diagnostics = {}
    for old in published["seeds"]:
        seed = str(old["seed"])
        diagnostics[seed] = {}
        for scope in scopes:
            diagnostics[seed][scope] = {}
            for method, field in (("strict_decoder", "builtin_decoder_exact4"), ("frozen_probe", "frozen_v6_train_probe_retention")):
                reference = old["safe_arm"][field]["scopes"][scope]["ordered_child_cosine"]
                value = summaries[seed][scope]["methods"][method]["ordered"]["mean"]
                diagnostics[seed][scope][method] = {"original": reference, "regenerated": value, "difference": value - reference,
                    "within_fixed_diagnostic_tolerance": abs(value - reference) <= cfg["historical_mean_diagnostic_tolerance"]}
    documents = [{"id": group, "corpus": payload["labels"][i],
                  "scores": {seed: {method: {metric: float(values[i]) for metric, values in scores[method].items()} for method in METHODS} for seed, scores in by_seed.items()},
                  "contrasts": {seed: {name: float(scores[a]["ordered"][i] - scores[b]["ordered"][i]) for name, (a, b) in CONTRASTS.items()} for seed, scores in by_seed.items()},
                  "geometry": {seed: {name: float(values[i]) for name, values in scores.items()} for seed, scores in geometry.items()}}
                 for i, group in enumerate(payload["groups"])]
    atomic_json(root / "CROSS_COSINES.json", {"groups": payload["groups"], "labels": payload["labels"], "matrices": {key: value.tolist() for key, value in matrices.items()}})
    atomic_json(root / "DOCUMENT_SCORES.json", documents)
    atomic_json(root / "RESULTS.json", {"study": cfg["study"], "configuration": cfg, "documents": 606,
        "scope_order": list(scopes), "summaries": summaries,
        "geometry": {seed: {scope: {key: summarize_scores(value[mask]) for key, value in values.items()} for scope, mask in scopes.items()} for seed, values in geometry.items()},
        "historical_reproduction_diagnostics": diagnostics,
        "source_sha256": manifest["source_sha256"], "immutable_input_sha256": manifest["immutable_input_sha256"],
        "inference_manifest_sha256": sha256(root / "INFERENCE_MANIFEST.json"), "document_scores_sha256": sha256(root / "DOCUMENT_SCORES.json"),
        "cross_cosines_sha256": sha256(root / "CROSS_COSINES.json"), "population_order_sha256": stable_hash(payload["groups"]),
        "no_matched_pooling_probe_replay": True, "completed_unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "analyse", "all"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage in ("infer", "all"):
        infer(args.config, args.device)
    if args.stage in ("analyse", "all"):
        analyse(args.config)
