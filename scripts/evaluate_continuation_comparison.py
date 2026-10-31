#!/usr/bin/env python3
"""Compare each released continuation source with its frozen clean-v6 final model."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

from evaluate_reconstruction_calibration import load_protocol as load_calibration_protocol, load_inference as load_calibration_inference, predict
from evaluate_safe_baselines import load_inference as load_safe_inference, padded_batches, attack_result
from safe_baselines_protocol import ROOT, atomic_json, atomic_tensor, configure_device, environment, load_safe_cache, require_hash, sha256, stable_hash
from recursive_geodesic_composition.evaluate_kary_validation import build_model, ParentOnlyRetentionProbe
from recursive_geodesic_composition.safe_baselines import cosine_rows, validate_vectors
from recursive_geodesic_composition.reconstruction_calibration import cross_cosines, assignment_scores
from recursive_geodesic_composition.continuation_comparison import paired_summary, paired_auc

CONFIG = ROOT / "configs/continuation_comparison_v1.yaml"
ROLES = {"source": "continuation_initialization_source_checkpoint", "final": "final_clean_v6_checkpoint",
         "source_probe": "source_parent_only_fitted_probe", "final_probe": "candidate_parent_only_fitted_probe"}


def load_protocol(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["study"] != "continuation_comparison_v1" or cfg["output_dir"] != "results/continuation_comparison_v1":
        raise ValueError("wrong additive namespace")
    if cfg["seeds"] != [424242, 271828, 314159] or cfg["stages"] != ["source", "final"] or cfg["expected_safe_documents"] != 4000 or cfg["expected_probe_documents"] != 606:
        raise ValueError("population/checkpoint pairing changed")
    if any(cfg[key] != 0 for key in ("merger_updates", "decoder_updates", "probe_updates", "teacher_encoding_calls", "final_classifier_fits")):
        raise ValueError("no model updates/teacher encoding/final classifier refits are permitted")
    if cfg["source_classifier_fits"] != 30 or cfg["bootstrap"]["reps"] != 2000:
        raise ValueError("classifier/bootstrap protocol changed")
    require_hash(ROOT / cfg["calibration_config"], cfg["calibration_config_sha256"])
    for file, digest in cfg["immutable_inputs"].items():
        require_hash(ROOT / file, digest)
    cfg["config_sha256"] = sha256(path)
    calibration, _, baseline, official, published = load_calibration_protocol(ROOT / cfg["calibration_config"])
    return cfg, calibration, baseline, official, published


def source_hashes():
    files = ["configs/continuation_comparison_v1.yaml", "scripts/evaluate_continuation_comparison.py",
             "scripts/render_continuation_comparison.py", "scripts/audit_continuation_comparison.py",
             "src/recursive_geodesic_composition/continuation_comparison.py",
             "src/recursive_geodesic_composition/reconstruction_calibration.py",
             "src/recursive_geodesic_composition/safe_baselines.py", "scripts/safe_baselines_protocol.py",
             "scripts/evaluate_safe_baselines.py", "scripts/evaluate_reconstruction_calibration.py",
             "scripts/official_v6/evaluate_official_test_v6_recovery.py",
             "src/recursive_geodesic_composition/architecture.py",
             "src/recursive_geodesic_composition/evaluate_kary_validation.py",
             "scripts/final_v6/freeze_training_and_probes.py"]
    return {path: sha256(ROOT / path) for path in files}


def artifacts(baseline):
    release = json.loads((ROOT / baseline["release_manifest"]).read_text())
    result = {}
    for seed in baseline["seeds"]:
        result[str(seed)] = {}
        for label, role in ROLES.items():
            candidates = [row for row in release["artifacts"] if row["seed"] == seed and row["role"] == role]
            if len(candidates) != 1:
                raise ValueError("release must identify one artifact per role/seed")
            entry = candidates[0] | {"path": str(Path(baseline["artifact_dir"]) / candidates[0]["asset_filename"])}
            require_hash(ROOT / entry["path"], entry["sha256"])
            if (ROOT / entry["path"]).stat().st_size != entry["byte_size"]:
                raise ValueError("artifact byte count mismatch")
            result[str(seed)][label] = entry
    return result


def inference_signature(cfg):
    return {key: cfg[key] for key in ("arity", "encoder_hidden", "encoder_layers", "heads", "importance_mode", "geodesic_parameterization")} | {
        "max_angle": float(cfg.get("max_geodesic_angle", cfg.get("max_angle", .8))),
        "importance_beta": float(cfg.get("importance_beta", 1.))}


def verify_lineage(source, final, source_entry):
    stage = final["staged_training"]
    if stage["initialization_checkpoint_sha256"] != source_entry["sha256"] or stage["initialization_checkpoint"] != source_entry["protocol_path"]:
        raise ValueError("final checkpoint does not continue from this source")
    if source["seed"] != final["seed"] or final["selected_step"] != 400 or stage["freeze_encoder_steps"] != 300 or not stage["continued_autoencoder_retraining"]:
        raise ValueError("continuation identity/schedule mismatch")
    if inference_signature(source["config"]) != inference_signature(final["config"]):
        raise ValueError("source/final inference architecture differs")
    a, b = source["state_dict"], final["state_dict"]
    keys = [key for key in a if not key.startswith("strict_decoder.")]
    if set(keys) != {key for key in b if not key.startswith("strict_decoder.")} or any(a[key].shape != b[key].shape for key in keys):
        raise ValueError("merger state layout differs")
    changed = sum(not torch.equal(a[key], b[key]) for key in keys)
    if changed != len(keys):
        raise ValueError("not every merger tensor changed as recorded by the original freeze")
    return {"staged_training": stage, "same_inference_signature": inference_signature(source["config"]),
            "merger_tensors": len(keys), "changed_merger_tensors": changed,
            "merger_parameters": sum(a[key].numel() for key in keys), "not_training_from_scratch": True}


def load_source_probe(entry, candidate_metadata, device):
    require_hash(ROOT / entry["path"], entry["sha256"])
    checkpoint = torch.load(ROOT / entry["path"], map_location="cpu", weights_only=True)
    if (checkpoint["seed"], checkpoint["arm"], checkpoint["fit_role"]) != (entry["seed"], "source", "fresh_v6_train"):
        raise ValueError("wrong source probe")
    for key in ("selection_audit", "schedule_sha256", "seed", "fit_role"):
        if checkpoint[key] != candidate_metadata[key]:
            raise ValueError("source/candidate TRAIN probe metadata is not matched")
    model = ParentOnlyRetentionProbe(768, 4, 512).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), {key: value for key, value in checkpoint.items() if key != "state_dict"}


def inputs(calibration, baseline, published):
    safe, safe_manifest = load_safe_inference(baseline)
    cal, cal_manifest = load_calibration_inference(calibration)
    rows, _ = load_safe_cache(baseline, published)
    require_hash(ROOT / baseline["cache_dir"] / "CACHE_MANIFEST.json", safe_manifest["cache_manifest_sha256"])
    if safe["groups"].tolist() != [row["id"] for row in rows]:
        raise ValueError("safe input row identities differ")
    exact = np.flatnonzero(safe["active_children"] == 4)
    if safe["groups"][exact].tolist() != cal["groups"] or not torch.equal(torch.stack([rows[i]["leaves"] for i in exact]), cal["children"]):
        raise ValueError("exact-four probe targets differ")
    for seed in baseline["seeds"]:
        if not np.array_equal(safe["vectors"][f"learned_parent_seed{seed}"][exact], cal["parents"][str(seed)].numpy()):
            raise ValueError("final probe and teacher-fidelity parents differ")
    return safe, cal, rows, exact, safe_manifest, cal_manifest


def infer(config_path, device_name):
    cfg, calibration, baseline, _, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    root = ROOT / cfg["output_dir"]
    if root.exists():
        raise FileExistsError("create-only inference; use analyse/report/audit for saved outputs")
    safe, cal, rows, exact, safe_manifest, cal_manifest = inputs(calibration, baseline, published)
    entries = artifacts(baseline)
    immutable = cal_manifest["immutable_input_sha256"] | cfg["immutable_inputs"] | {cfg["calibration_config"]: cfg["calibration_config_sha256"]}
    freeze = {"configuration": cfg, "source_sha256": source_hashes(), "immutable_input_sha256": immutable,
              "environment": environment(device), "artifacts": entries, "started_unix": time.time(),
              "cache_manifest_sha256": safe_manifest["cache_manifest_sha256"]}
    root.mkdir(parents=True)
    atomic_json(root / "ANALYSIS_PROTOCOL.json", freeze)
    batches, indices = padded_batches(rows, device, cfg["inference"]["batch_size"])
    source_parents, source_probes, lineage, probe_metadata = {}, {}, {}, {}
    with torch.inference_mode():
        for seed in cfg["seeds"]:
            key, entry = str(seed), entries[str(seed)]
            model, source = build_model(ROOT / entry["source"]["path"], device)
            final = torch.load(ROOT / entry["final"]["path"], map_location="cpu", weights_only=True)
            lineage[key] = verify_lineage(source, final, entry["source"])
            del source, final
            roots = torch.empty(4000, 768)
            for (x, n), index in zip(batches, indices):
                roots[index] = model.merge(x, n).cpu()
            validate_vectors(roots.numpy(), unit=True)
            source_parents[key] = roots
            del model
            probe, probe_metadata[key] = load_source_probe(entry["source_probe"], cal_manifest["probe_training_provenance"][key], device)
            source_probes[key] = predict(probe, roots[exact], device, cfg["inference"]["batch_size"])
            validate_vectors(source_probes[key].reshape(-1, 768).numpy(), unit=True)
            del probe
            for item in entry.values():
                require_hash(ROOT / item["path"], item["sha256"])
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Source merger and matched source probe completed: {seed}", flush=True)
    payload = {"source_parents": source_parents, "source_probes": source_probes,
               "final_parents": {str(seed): torch.from_numpy(safe["vectors"][f"learned_parent_seed{seed}"]) for seed in cfg["seeds"]},
               "final_probes": {str(seed): cal["predictions"][f"frozen_probe_seed{seed}"] for seed in cfg["seeds"]},
               "weighted_parents": torch.from_numpy(safe["vectors"]["token_weighted_mean"]),
               "targets": torch.from_numpy(safe["targets"]), "children": cal["children"],
               "groups": safe["groups"].tolist(), "labels": safe["labels"].tolist(), "active_children": safe["active_children"].tolist(),
               "exact_four_indices": exact.tolist(), "probe_groups": cal["groups"]}
    atomic_tensor(root / "vectors.pt", payload)
    for path, digest in immutable.items():
        require_hash(ROOT / path, digest)
    atomic_json(root / "INFERENCE_MANIFEST.json", freeze | {"vectors_sha256": sha256(root / "vectors.pt"),
        "population_order_sha256": stable_hash(payload["groups"]), "lineage": lineage, "source_probe_metadata": probe_metadata,
        "final_predictions_reused_without_change": True, "new_model_updates": 0, "new_teacher_encoding_calls": 0,
        "completed_unix": time.time()})


def load_inference(cfg):
    root = ROOT / cfg["output_dir"]
    m = json.loads((root / "INFERENCE_MANIFEST.json").read_text())
    if m["configuration"]["config_sha256"] != cfg["config_sha256"]:
        raise ValueError("configuration changed")
    for section in ("source_sha256", "immutable_input_sha256"):
        for file, digest in m[section].items():
            require_hash(ROOT / file, digest)
    require_hash(root / "vectors.pt", m["vectors_sha256"])
    payload = torch.load(root / "vectors.pt", map_location="cpu", weights_only=True)
    if stable_hash(payload["groups"]) != m["population_order_sha256"]:
        raise ValueError("output document order changed")
    return payload, m


def score_outputs(payload, cfg):
    exact, children = np.asarray(payload["exact_four_indices"]), payload["children"].numpy()
    teacher, retention, matrices = {}, {}, {}
    weighted = cosine_rows(payload["weighted_parents"].numpy(), payload["targets"].numpy())
    for seed in map(str, cfg["seeds"]):
        src, fin = (payload[f"{stage}_parents"][seed].numpy() for stage in ("source", "final"))
        a, b = (cosine_rows(parent, payload["targets"].numpy()) for parent in (src, fin))
        teacher[seed] = {"source": a, "final": b, "delta": b - a, "output_agreement": cosine_rows(src, fin),
                         "weighted": weighted, "source_minus_weighted": a - weighted, "final_minus_weighted": b - weighted}
        r = {}
        for stage, parents in (("source", src), ("final", fin)):
            matrix = cross_cosines(payload[f"{stage}_probes"][seed].numpy(), children)
            copied = cross_cosines(np.repeat(parents[exact, None], 4, axis=1), children)
            matrices[f"{stage}_probe_seed{seed}"] = matrix
            matrices[f"{stage}_copy_seed{seed}"] = copied
            scores, copy = assignment_scores(matrix), assignment_scores(copied)["ordered"]
            r[f"{stage}_probe"] = scores["ordered"]
            r[f"{stage}_copy"] = copy
            r[f"{stage}_gain"] = scores["ordered"] - copy
            r[f"{stage}_order_gap"] = scores["ordered_minus_shuffled"]
        for metric in ("probe", "copy", "gain", "order_gap"):
            r[f"{metric}_delta"] = r[f"final_{metric}"] - r[f"source_{metric}"]
        retention[seed] = r
    for scores in (teacher, retention):
        scores["three_seed_mean"] = {key: np.mean([scores[str(seed)][key] for seed in cfg["seeds"]], axis=0) for key in scores[str(cfg["seeds"][0])]}
    return teacher, retention, matrices


def saved_final_attack(baseline, result, seed, scope):
    relative = f"attacks/learned_parent_seed{seed}__{scope}.json"
    path = ROOT / baseline["output_dir"] / relative
    require_hash(path, result["attack_result_sha256"][relative])
    entry = json.loads(path.read_text())
    if entry["attack_recipe_sha256"] != baseline["attack_config_sha256"] or entry["inference_manifest_sha256"] != result["inference_manifest_sha256"]:
        raise ValueError("cached final classifier/input recipe differs")
    return entry


def analyse(config_path):
    cfg, _, baseline, official, published = load_protocol(config_path)
    root = ROOT / cfg["output_dir"]
    if (root / "RESULTS.json").exists():
        raise FileExistsError("completed evaluation is create-only")
    payload, manifest = load_inference(cfg)
    teacher, retention, matrices = score_outputs(payload, cfg)
    labels, exact = np.asarray(payload["labels"]), np.asarray(payload["exact_four_indices"])
    scopes = {"pooled": np.ones(4000, dtype=bool)} | {c: labels == c for c in published["corpus_order"]}
    descriptive = scopes | {f"arity_{k}": np.asarray(payload["active_children"]) == k for k in (2, 3, 4)}
    reps, seed = cfg["bootstrap"]["reps"], cfg["bootstrap"]["paired_score_seed"]
    fidelity = {s: {scope: paired_summary(scores, mask, reps, seed + i) for i, (scope, mask) in enumerate(descriptive.items())} for s, scores in teacher.items()}
    reconstruction = {s: {scope: paired_summary(scores, mask[exact], reps, seed + i) for i, (scope, mask) in enumerate(scopes.items())} for s, scores in retention.items()}
    atomic_json(root / "POINTS.json", {"groups": payload["groups"], "labels": payload["labels"], "active_children": payload["active_children"],
        "exact_four_indices": payload["exact_four_indices"], "probe_groups": payload["probe_groups"],
        "teacher": {s: {k: v.tolist() for k, v in scores.items()} for s, scores in teacher.items()},
        "reconstruction": {s: {k: v.tolist() for k, v in scores.items()} for s, scores in retention.items()}})
    atomic_json(root / "PROBE_CROSS_COSINES.json", {"groups": payload["probe_groups"], "matrices": {k: v.tolist() for k, v in matrices.items()}})
    previous = json.loads((ROOT / baseline["output_dir"] / "RESULTS.json").read_text())
    adapter = baseline | {"output_dir": cfg["output_dir"], "config_sha256": cfg["config_sha256"]}
    attacks, auc, final_hashes = {}, {}, {}
    with threadpool_limits(limits=cfg["inference"]["cpu_threads"]):
        for i, (scope, mask) in enumerate(scopes.items()):
            source_bundles, final_bundles, families = [], [], {}
            for model_seed in cfg["seeds"]:
                source = attack_result(f"source_seed{model_seed}", scope, payload["targets"].numpy()[mask], payload["source_parents"][str(model_seed)].numpy()[mask],
                    np.asarray(payload["groups"])[mask], adapter, official, published, i)
                final = saved_final_attack(baseline, previous, model_seed, scope)
                for field in ("classifier_train_groups", "classifier_holdout_groups"):
                    if source[field] != final[field]:
                        raise ValueError("source/final classifier split differs")
                families[str(model_seed)] = {}
                for stage, cell, bundles in (("source", source, source_bundles), ("final", final, final_bundles)):
                    family = cell["metrics"]["strongest_attack"]
                    families[str(model_seed)][stage] = family
                    bundles.append(cell["bundles"][family])
                final_path = f"{baseline['output_dir']}/attacks/learned_parent_seed{model_seed}__{scope}.json"
                final_hashes[final_path] = sha256(ROOT / final_path)
            auc[scope] = paired_auc(source_bundles, final_bundles, reps, cfg["bootstrap"]["paired_auc_seed"] + i) | {"selected_families": families}
            attacks[scope] = {"source": source_bundles, "final": final_bundles}
            if abs(auc[scope]["three_seed_mean"]["final"]["mean"] - previous["detectability"]["learned_parent"][scope]["mean_auc"]) > cfg["reference_score_tolerance"]:
                raise ValueError("cached final AUC point changed")
    atomic_json(root / "SELECTED_CLASSIFIER_SCORES.json", attacks)
    for path, digest in manifest["immutable_input_sha256"].items():
        require_hash(ROOT / path, digest)
    atomic_json(root / "RESULTS.json", {"study": cfg["study"], "configuration": cfg, "scope_order": list(scopes), "fidelity_scope_order": list(descriptive),
        "teacher": fidelity, "reconstruction": reconstruction, "source_detection": auc,
        "source_sha256": manifest["source_sha256"], "immutable_input_sha256": manifest["immutable_input_sha256"],
        "inference_manifest_sha256": sha256(root / "INFERENCE_MANIFEST.json"), "points_sha256": sha256(root / "POINTS.json"),
        "probe_cross_cosines_sha256": sha256(root / "PROBE_CROSS_COSINES.json"), "selected_classifier_scores_sha256": sha256(root / "SELECTED_CLASSIFIER_SCORES.json"),
        "source_attack_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "attacks").glob("*.json"))},
        "reused_final_attack_sha256": final_hashes, "source_classifier_fits": 30, "final_classifier_fits": 0,
        "not_an_objective_isolating_ablation": True, "completed_unix": time.time()})


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
