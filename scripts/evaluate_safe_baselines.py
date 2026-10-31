#!/usr/bin/env python3
"""Additive post-review safe-arm baselines; never invoke the official run entry point."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

from safe_baselines_protocol import (
    DEFAULT_CONFIG, ROOT, PUBLISHED_HASH, REPLAY_ID, atomic_json, atomic_tensor,
    checkpoint_entries, configure_device, environment, load_protocol, load_safe_cache,
    replay_source_hashes, require_hash, sha256, source_hashes, stable_hash,
)
from public_replay import package_file_hash

sys.path.insert(0, str(ROOT / "src"))

from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.safe_baselines import (
    BASELINES, LEARNED, CONTRASTS, baseline_parent, benchmark, cosine_rows,
    fidelity_summary, shuffled_cosines, teacher_permutations, validate_children,
    validate_vectors,
)

# Reuse the original pure classifier/split/bootstrap functions, not its entry
# point, cache loader, reporting gate, or one-shot markers.
sys.path.insert(0, str(ROOT / "scripts/official_v6"))
from evaluate_official_test_v6_recovery import (
    evaluate_fixed_attacks, grouped_attack_split, three_seed_shared_ci,
)

LABELS = {
    "first_chunk": "First chunk", "unweighted_mean": "Unweighted mean",
    "token_weighted_mean": "Token-weighted mean", "learned_reference": "Learned reference",
    "learned_parent": "Final learned parent",
}


def replay_checkpoint_identity(entry):
    return {key: entry[key] for key in ("seed", "role", "path", "asset_filename", "byte_size", "sha256")}


def document_metadata(rows):
    return [{key: row[key] for key in (
        "id", "corpus", "source_index", "safe_prefix_token_count", "full_token_count",
        "safe_prefix_token_ids_sha256")} | {"active_children": len(row["lengths"])} for row in rows]


def expected_vector_keys(cfg):
    return set(BASELINES) | {f"{method}_seed{seed}" for method in LEARNED for seed in cfg["seeds"]}


def padded_batches(rows, device, batch_size):
    children = torch.zeros(len(rows), 4, 768, dtype=torch.float32)
    lengths = torch.zeros(len(rows), 4, dtype=torch.float32)
    for index, row in enumerate(rows):
        count = len(row["lengths"])
        children[index, :count] = row["leaves"]
        lengths[index, :count] = row["lengths"]
    validate_children(children, lengths)
    batches, indices = [], []
    # Preserve the original direct evaluator's arity grouping and batch size.
    for arity in (2, 3, 4):
        group = torch.nonzero((lengths > 0).sum(1) == arity, as_tuple=True)[0]
        for start in range(0, len(group), batch_size):
            selection = group[start:start + batch_size]
            indices.append(selection.numpy())
            batches.append((children[selection].to(device), lengths[selection].to(device)))
    return batches, indices


def infer(config_path, device_name):
    cfg, official, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    output = ROOT / cfg["output_dir"]
    manifest_path = output / "INFERENCE_MANIFEST.json"
    if manifest_path.exists():
        load_inference(cfg)
        print("Verified existing inference; not recomputing.", flush=True)
        return
    if cfg.get("execution_mode") == REPLAY_ID and output.exists() and any(output.iterdir()):
        raise FileExistsError("incomplete or contradictory replay inference exists; use a fresh output namespace")
    rows, cache_manifest = load_safe_cache(cfg, published)
    entries = checkpoint_entries(cfg)
    output.mkdir(parents=True, exist_ok=True)
    replay = cfg.get("execution_mode") == REPLAY_ID
    sources = replay_source_hashes() if replay else source_hashes()
    runtime = environment(device)
    if replay:
        runtime.pop("git_commit", None)
        atomic_json(output / "INPUT_VIEW.json", cache_manifest)
    protocol = {
        "configuration": cfg, "source_sha256": sources,
        "environment": runtime, "started_unix": time.time(),
        "analysis_role": cfg["analysis_role"], "original_official_results_unchanged": True,
        "cache_manifest_sha256": sha256(ROOT / cfg["cache_dir"] / "CACHE_MANIFEST.json"),
    }
    if replay:
        protocol.update({"execution_mode": REPLAY_ID,
                         "input_view_sha256": sha256(output / "INPUT_VIEW.json"),
                         "teacher_encoding_calls": 0})
    atomic_json(output / "ANALYSIS_PROTOCOL.json", protocol)
    batches, indices = padded_batches(rows, device, int(cfg["inference"]["batch_size"]))
    vectors, timing, model_provenance = {}, {}, []
    warmup, repeats = cfg["timing"]["warmup_passes"], cfg["timing"]["measured_passes"]
    with torch.inference_mode():
        for method in BASELINES:
            result = torch.empty(len(rows), 768)
            for (children, lengths), selection in zip(batches, indices):
                result[selection] = baseline_parent(children, lengths, method).cpu()
            validate_vectors(result.numpy(), method, unit=True)
            vectors[method] = result
            timing[method] = benchmark(lambda x, n: baseline_parent(x, n, method), batches,
                                       device, warmup, repeats, len(rows))
        for entry in entries:
            seed = entry["seed"]
            model, checkpoint = build_model(ROOT / entry["path"], device)
            if checkpoint.get("seed") != seed or model.arity != 4 or model.d != 768:
                raise ValueError("checkpoint identity/architecture mismatch")
            roots, references = torch.empty(len(rows), 768), torch.empty(len(rows), 768)
            for (children, lengths), selection in zip(batches, indices):
                parent, aux = model.merge(children, lengths, return_aux=True)
                roots[selection] = parent.cpu()
                references[selection] = aux["anchor"].cpu()
            for method, values in (("learned_parent", roots), ("learned_reference", references)):
                key = f"{method}_seed{seed}"
                validate_vectors(values.numpy(), key, unit=True)
                vectors[key] = values
            measured = benchmark(lambda x, n: model.merge(x, n, return_aux=True), batches,
                                 device, warmup, repeats, len(rows))
            # This is a single measured full-forward path returning both values,
            # not a claim about the speed of an optimized reference-only model.
            timing[f"learned_parent_seed{seed}"] = measured | {"path": "full_merge_returning_parent_and_reference"}
            timing[f"learned_reference_seed{seed}"] = measured | {"path": "full_forward_diagnostic_extraction", "same_measurement_as_parent": True}
            require_hash(ROOT / entry["path"], entry["sha256"])
            model_provenance.append(replay_checkpoint_identity(entry) if replay else entry)
            print(f"Inferred and timed checkpoint {seed}", flush=True)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
    atomic_json(output / "DOCUMENT_METADATA.json", document_metadata(rows))
    payload = {
        "vectors": vectors, "targets": torch.stack([row["teacher_target"] for row in rows]),
        "groups": [row["id"] for row in rows], "labels": [row["corpus"] for row in rows],
        "active_children": [len(row["lengths"]) for row in rows],
    }
    vector_path = output / "vectors.pt"
    atomic_tensor(vector_path, payload)
    manifest = {
        "study": cfg["study"], "analysis_role": cfg["analysis_role"],
        "config_sha256": cfg["config_sha256"], "source_sha256": sources,
        "environment": runtime, "checkpoints": model_provenance,
        "cache_manifest_sha256": sha256(ROOT / cfg["cache_dir"] / "CACHE_MANIFEST.json"),
        "cache_regenerated": not replay, "original_embedding_bytes_claimed": False,
        "population_order_sha256": stable_hash(payload["groups"]),
        "vectors_sha256": sha256(vector_path), "metadata_sha256": sha256(output / "DOCUMENT_METADATA.json"),
        "timing_protocol": cfg["timing"], "timings": timing,
        "batch_size": cfg["inference"]["batch_size"], "dtype": "float32",
        "documents": len(rows), "merger_updates": 0, "completed_unix": time.time(),
    }
    if replay:
        for name, digest in sources.items():
            require_hash(ROOT / name, digest)
        for entry in cache_manifest["official_safe_files"].values():
            require_hash(ROOT / entry["official_safe_path"], entry["official_safe_sha256"])
        manifest.update({
            "execution_mode": REPLAY_ID,
            "input_view_sha256": sha256(output / "INPUT_VIEW.json"),
            "analysis_protocol_sha256": sha256(output / "ANALYSIS_PROTOCOL.json"),
            "official_safe_tensor_bytes_from_public_release": True,
            "historical_regenerated_cache_recovered": False,
            "teacher_encoding_calls": 0,
            "timing_interpretation": "current hardware only; not a reproduction or comparison of archived GPU timing",
        })
    atomic_json(manifest_path, manifest)
    require_hash(ROOT / cfg["published_results"], PUBLISHED_HASH)


def load_inference(cfg):
    output = ROOT / cfg["output_dir"]
    manifest = json.loads((output / "INFERENCE_MANIFEST.json").read_text())
    if manifest["config_sha256"] != cfg["config_sha256"]:
        raise ValueError("inference protocol changed")
    require_hash(output / "vectors.pt", manifest["vectors_sha256"])
    require_hash(output / "DOCUMENT_METADATA.json", manifest["metadata_sha256"])
    payload = torch.load(output / "vectors.pt", map_location="cpu", weights_only=True)
    if stable_hash(payload["groups"]) != manifest["population_order_sha256"]:
        raise ValueError("inference document order changed")
    if cfg.get("execution_mode") == REPLAY_ID:
        validate_replay_inference(cfg, payload, manifest)
    payload["vectors"] = {key: validate_vectors(value.numpy(), key, unit=True)
                          for key, value in payload["vectors"].items()}
    payload["targets"] = validate_vectors(payload["targets"].numpy(), "teacher", unit=True)
    for key in ("groups", "labels", "active_children"):
        payload[key] = np.asarray(payload[key])
    return payload, manifest


def validate_replay_inference(cfg, payload, manifest):
    output = ROOT / cfg["output_dir"]
    if (manifest.get("execution_mode") != REPLAY_ID or manifest.get("documents") != 4000 or
            manifest.get("merger_updates") != 0 or manifest.get("teacher_encoding_calls") != 0 or
            manifest.get("historical_regenerated_cache_recovered") is not False or
            manifest.get("cache_regenerated") is not False or
            manifest.get("timing_protocol") != cfg["timing"] or
            manifest.get("batch_size") != cfg["inference"]["batch_size"] or
            manifest.get("dtype") != "float32"):
        raise ValueError("incomplete or contradictory replay inference manifest")
    if manifest["source_sha256"] != replay_source_hashes():
        raise ValueError("replay inference implementation changed")
    require_hash(output / "INPUT_VIEW.json", manifest["input_view_sha256"])
    require_hash(output / "ANALYSIS_PROTOCOL.json", manifest["analysis_protocol_sha256"])
    protocol = json.loads((output / "ANALYSIS_PROTOCOL.json").read_text())
    if (protocol.get("configuration") != cfg or protocol.get("execution_mode") != REPLAY_ID or
            protocol.get("source_sha256") != manifest["source_sha256"] or
            protocol.get("environment") != manifest["environment"] or
            protocol.get("input_view_sha256") != manifest["input_view_sha256"] or
            protocol.get("cache_manifest_sha256") != manifest["cache_manifest_sha256"]):
        raise ValueError("replay inference freeze disagrees with its manifest")
    expected_checkpoints = [replay_checkpoint_identity(entry) for entry in checkpoint_entries(cfg)]
    if manifest["checkpoints"] != expected_checkpoints:
        raise ValueError("replay checkpoint identities changed")
    published = json.loads((ROOT / cfg["published_results"]).read_text())
    require_hash(ROOT / cfg["published_results"], PUBLISHED_HASH)
    rows, view = load_safe_cache(cfg, published)
    if (view != json.loads((output / "INPUT_VIEW.json").read_text()) or
            view["source_official_manifest_sha256"] != manifest["cache_manifest_sha256"]):
        raise ValueError("replay public input view changed after inference")
    metadata = json.loads((output / "DOCUMENT_METADATA.json").read_text())
    if metadata != document_metadata(rows):
        raise ValueError("replay document metadata differs from the verified input view")
    if (payload["groups"] != [row["id"] for row in rows] or
            payload["labels"] != [row["corpus"] for row in rows] or
            payload["active_children"] != [len(row["lengths"]) for row in rows]):
        raise ValueError("replay vector population or order differs from the input view")
    targets = payload["targets"]
    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.float32 or
            targets.shape != (4000, 768) or
            not torch.equal(targets, torch.stack([row["teacher_target"] for row in rows]))):
        raise ValueError("replay teacher targets are not the released official targets")
    keys = expected_vector_keys(cfg)
    if set(payload["vectors"]) != keys or set(manifest["timings"]) != keys:
        raise ValueError("replay omitted a method, checkpoint, or timing")
    for key, value in payload["vectors"].items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != (4000, 768):
            raise ValueError(f"{key}: replay vector shape or dtype changed")
        duration = manifest["timings"][key]["median_ms_per_document"]
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"{key}: invalid current-hardware timing")


def attack_result(key, scope, real, fake, groups, cfg, official, published, scope_index):
    output = ROOT / cfg["output_dir"]
    path = output / "attacks" / f"{key}__{scope}.json"
    attack_cfg = yaml.safe_load((ROOT / cfg["attack_config"]).read_text())
    _, _, paired, train, held = grouped_attack_split(real, fake, groups, attack_cfg)
    train_ids = sorted(map(str, np.unique(paired[train])))
    held_ids = sorted(map(str, np.unique(paired[held])))
    expected = published["seeds"][0]["safe_arm"]["scopes"][scope]["attacks"]
    if stable_hash(train_ids) != expected["train_group_sha256"] or stable_hash(held_ids) != expected["holdout_group_sha256"]:
        raise ValueError(f"classifier document split differs from published split: {scope}")
    inference_hash = sha256(output / "INFERENCE_MANIFEST.json")
    if path.exists():
        result = json.loads(path.read_text())
        if result["inference_manifest_sha256"] != inference_hash or result["config_sha256"] != cfg["config_sha256"]:
            raise ValueError("cached classifier result has incompatible input/protocol")
        return result
    seed = int(key.rsplit("_seed", 1)[1]) if "_seed" in key else None
    bootstrap_seed = (int(official["bootstrap"]["per_seed_seeds"][seed]) if seed is not None
                      else int(cfg["bootstrap"]["baseline_attack_seed"])) + 100 * scope_index
    print(f"Fitting fixed classifiers: {key} / {scope}", flush=True)
    metrics, bundles = evaluate_fixed_attacks(real, fake, groups, attack_cfg,
                                             int(cfg["bootstrap"]["reps"]), bootstrap_seed)
    result = {
        "representation": key, "scope": scope, "metrics": metrics,
        "classifier_train_groups": train_ids, "classifier_holdout_groups": held_ids,
        "bundles": {name: {field: values.tolist() for field, values in bundle.items()}
                    for name, bundle in bundles.items()},
        "inference_manifest_sha256": inference_hash, "config_sha256": cfg["config_sha256"],
        "attack_recipe_sha256": cfg["attack_config_sha256"], "bootstrap_seed": bootstrap_seed,
    }
    atomic_json(path, result)
    return result


def compare_archived_baselines(cfg, groups, labels, active_children, raw_scores, fidelity, attacks):
    results_path = "results/safe_baselines_v1/RESULTS.json"
    documents_path = "results/safe_baselines_v1/DOCUMENT_SCORES.json"
    results_hash = package_file_hash(ROOT, results_path)
    documents_hash = package_file_hash(ROOT, documents_path)
    archived = json.loads((ROOT / results_path).read_text())
    documents = json.loads((ROOT / documents_path).read_text())
    if (archived["configuration"]["config_sha256"] != cfg["scientific_config_sha256"] or
            archived["documents"] != 4000 or len(documents) != 4000 or
            archived["population_order_sha256"] != stable_hash(groups.tolist())):
        raise ValueError("archived baseline comparison population or scientific protocol changed")
    for index, row in enumerate(documents):
        if (row["id"] != groups[index] or row["corpus"] != labels[index] or
                row["active_children"] != int(active_children[index]) or
                set(row["cosines"]) != set(raw_scores)):
            raise ValueError("archived baseline per-document scores are not paired to this replay")
    masks = {"pooled": np.ones(len(groups), dtype=bool)}
    masks.update({corpus: labels == corpus for corpus in dict.fromkeys(labels)})
    masks.update({f"arity_{k}": active_children == k for k in (2, 3, 4)})
    document_differences = {}
    for key, values in raw_scores.items():
        old = np.asarray([row["cosines"][key] for row in documents], dtype=np.float64)
        if not np.isfinite(old).all():
            raise ValueError(f"{key}: nonfinite archived document scores")
        document_differences[key] = {}
        for scope, mask in masks.items():
            delta = values[mask] - old[mask]
            document_differences[key][scope] = {
                "documents": int(mask.sum()),
                "replay_mean": float(values[mask].mean()),
                "archived_mean": float(old[mask].mean()),
                "mean_signed_difference": float(delta.mean()),
                "mean_absolute_difference": float(np.abs(delta).mean()),
                "maximum_absolute_difference": float(np.abs(delta).max()),
            }
    fidelity_differences = {}
    for seed, scopes in fidelity.items():
        fidelity_differences[seed] = {}
        for scope, summary in scopes.items():
            fidelity_differences[seed][scope] = {}
            for method, new in summary["methods"].items():
                old = archived["fidelity"][seed][scope]["methods"][method]
                fidelity_differences[seed][scope][method] = {
                    "replay_mean": new["mean"], "archived_mean": old["mean"],
                    "difference": new["mean"] - old["mean"],
                    "replay_ci95": new["ci95"], "archived_ci95": old["ci95"],
                }
    detectability_differences = {}
    for method, scopes in attacks.items():
        detectability_differences[method] = {}
        for scope, new in scopes.items():
            old = archived["detectability"][method][scope]
            detectability_differences[method][scope] = {
                "replay_auc": new["mean_auc"], "archived_auc": old["mean_auc"],
                "difference": new["mean_auc"] - old["mean_auc"],
                "replay_ci95": new["ci95"], "archived_ci95": old["ci95"],
                "replay_selected_families": new["selected_families"],
                "archived_selected_families": old["selected_families"],
            }
    return {
        "comparison": "cached original official safe tensors versus historical regenerated-safe baseline results",
        "archived_results_path": results_path, "archived_results_sha256": results_hash,
        "archived_document_scores_path": documents_path, "archived_document_scores_sha256": documents_hash,
        "paired_document_identities_verified": True,
        "fidelity": fidelity_differences, "detectability": detectability_differences,
        "document_cosine_differences": document_differences,
        "timings_compared": False,
    }


def load_replay_results(cfg):
    """Load only a completed replay whose inputs and every output still match."""
    if cfg.get("execution_mode") != REPLAY_ID:
        raise ValueError("completed replay loader requires the cached-replay configuration")
    output = ROOT / cfg["output_dir"]
    marker = json.loads((output / "COMPLETE.json").read_text())
    if marker.get("execution_mode") != REPLAY_ID or marker.get("config_sha256") != cfg["config_sha256"]:
        raise ValueError("baseline replay completion identity changed")
    require_hash(output / "ARTIFACT_HASHES.json", marker["artifact_hashes_sha256"])
    require_hash(output / "RESULTS.json", marker["results_sha256"])
    require_hash(output / "INFERENCE_MANIFEST.json", marker["inference_manifest_sha256"])
    payload, inference = load_inference(cfg)
    result = json.loads((output / "RESULTS.json").read_text())
    if (result.get("execution_mode") != REPLAY_ID or result.get("configuration") != cfg or
            result.get("documents") != 4000 or result.get("merger_updates") != 0 or
            result.get("source_sha256") != inference["source_sha256"] or
            result.get("input_view_sha256") != inference["input_view_sha256"] or
            result.get("inference_manifest_sha256") != marker["inference_manifest_sha256"] or
            result.get("population_order_sha256") != inference["population_order_sha256"]):
        raise ValueError("baseline replay results contradict the completed inference")
    require_hash(output / "DOCUMENT_SCORES.json", result["document_scores_sha256"])
    comparison = result["historical_baseline_comparison"]
    for path_key, hash_key in (("archived_results_path", "archived_results_sha256"),
                               ("archived_document_scores_path", "archived_document_scores_sha256")):
        if package_file_hash(ROOT, comparison[path_key]) != comparison[hash_key]:
            raise ValueError("archived baseline comparison identity changed")
    if comparison["timings_compared"] is not False:
        raise ValueError("archived hardware timing must not be treated as reproduced")
    scopes = ["pooled", *dict.fromkeys(payload["labels"].tolist())]
    expected_attacks = {f"attacks/{key}__{scope}.json"
                        for key in expected_vector_keys(cfg) for scope in scopes}
    if set(result["attack_result_sha256"]) != expected_attacks:
        raise ValueError("completed baseline replay has missing or unexpected classifier results")
    expected = expected_attacks | {
        "ANALYSIS_PROTOCOL.json", "INPUT_VIEW.json", "INFERENCE_MANIFEST.json",
        "vectors.pt", "DOCUMENT_METADATA.json", "DOCUMENT_SCORES.json", "RESULTS.json",
        "REPORT.md", "safe_baselines_table.tex",
    }
    paths = list(output.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("completed baseline replay contains a symbolic link")
    actual = {str(path.relative_to(output)) for path in paths if path.is_file()}
    if actual != expected | {"ARTIFACT_HASHES.json", "COMPLETE.json"}:
        raise ValueError("completed baseline replay has missing or unexpected artifacts")
    hashes = json.loads((output / "ARTIFACT_HASHES.json").read_text())
    if set(hashes) != expected:
        raise ValueError("baseline replay artifact inventory changed")
    for name, digest in hashes.items():
        require_hash(output / name, digest)
    for name, digest in result["attack_result_sha256"].items():
        if hashes[name] != digest:
            raise ValueError("baseline classifier result identity changed")
    return payload, inference, result


def evaluate(config_path):
    cfg, official, published = load_protocol(config_path)
    output = ROOT / cfg["output_dir"]
    replay = cfg.get("execution_mode") == REPLAY_ID
    if (output / "RESULTS.json").exists():
        if replay:
            load_replay_results(cfg)
            print("Verified completed baseline replay; not recomputing.", flush=True)
            return
        raise FileExistsError("complete analysis already exists; use report, not another evaluation")
    payload, inference_manifest = load_inference(cfg)
    if replay:
        expected = {"ANALYSIS_PROTOCOL.json", "INPUT_VIEW.json", "INFERENCE_MANIFEST.json",
                    "vectors.pt", "DOCUMENT_METADATA.json"}
        if {path.name for path in output.iterdir()} != expected:
            raise FileExistsError("incomplete or contradictory baseline analysis exists; use a fresh output namespace")
    vectors, targets = payload["vectors"], payload["targets"]
    groups, labels = payload["groups"], payload["labels"]
    if stable_hash(groups.tolist()) != published["cache_provenance"]["pooled_document_order_sha256"]:
        raise ValueError("evaluation population changed")
    raw_scores = {key: cosine_rows(values, targets) for key, values in vectors.items()}
    scopes = {"pooled": np.ones(len(groups), dtype=bool)}
    scopes.update({corpus: labels == corpus for corpus in published["corpus_order"]})
    descriptive_scopes = scopes | {f"arity_{k}": payload["active_children"] == k for k in (2, 3, 4)}
    reps, pair_seed = int(cfg["bootstrap"]["reps"]), int(cfg["bootstrap"]["paired_seed"])
    scores_by_seed, fidelity = {}, {}
    for seed in cfg["seeds"]:
        scores_by_seed[str(seed)] = {method: raw_scores[f"{method}_seed{seed}"] if method in LEARNED else raw_scores[method]
                                     for method in cfg["methods"]}
    # Average checkpoint scores within each source document, not normalized mean
    # embeddings and not three independent copies of the document population.
    scores_by_seed["three_seed_mean"] = {
        method: np.mean([scores_by_seed[str(seed)][method] for seed in cfg["seeds"]], axis=0)
        for method in cfg["methods"]
    }
    for name, scores in scores_by_seed.items():
        fidelity[name] = {scope: fidelity_summary(scores, mask, reps, pair_seed + offset)
                          for offset, (scope, mask) in enumerate(descriptive_scopes.items())}
    document_scores = []
    for index, group in enumerate(groups):
        document_scores.append({
            "id": str(group), "corpus": str(labels[index]),
            "active_children": int(payload["active_children"][index]),
            "cosines": {key: float(value[index]) for key, value in raw_scores.items()},
            "three_seed_mean_contrasts": {key: float(scores_by_seed["three_seed_mean"][a][index] - scores_by_seed["three_seed_mean"][b][index])
                                           for key, (a, b) in CONTRASTS.items()},
        })
    atomic_json(output / "DOCUMENT_SCORES.json", document_scores)
    attacks = {}
    for key, values in vectors.items():
        attacks[key] = {}
        for offset, (scope, mask) in enumerate(scopes.items()):
            attacks[key][scope] = attack_result(key, scope, targets[mask], values[mask], groups[mask],
                                               cfg, official, published, offset)
    aggregate_attacks = {}
    for method in cfg["methods"]:
        aggregate_attacks[method] = {}
        for scope in scopes:
            if method in BASELINES:
                metrics = attacks[method][scope]["metrics"]
                point = metrics["strongest_orientation_safe_auc"]
                ci = metrics["strongest_orientation_safe_auc_document_group_ci95"]
                families = [metrics["strongest_attack"]]
            else:
                bundles, families = [], []
                for seed in cfg["seeds"]:
                    entry = attacks[f"{method}_seed{seed}"][scope]
                    family = entry["metrics"]["strongest_attack"]
                    families.append(family)
                    bundles.append({field: np.asarray(value) for field, value in entry["bundles"][family].items()})
                point, ci = three_seed_shared_ci(bundles, reps, int(official["bootstrap"]["three_seed_shared_seeds"][scope]))
            aggregate_attacks[method][scope] = {"mean_auc": point, "ci95": ci, "selected_families": families}

    permutations = teacher_permutations(labels, int(cfg["calibration"]["permutations"]), int(cfg["calibration"]["seed"]))
    calibration = {}
    permutation_scores = {}
    for key, values in vectors.items():
        print(f"Shuffled-target calibration: {key}", flush=True)
        permutation_scores[key] = shuffled_cosines(values, targets, permutations)
    for method in cfg["methods"]:
        scores = permutation_scores[method] if method in BASELINES else np.mean(
            [permutation_scores[f"{method}_seed{seed}"] for seed in cfg["seeds"]], axis=0)
        calibration[method] = {}
        for scope, mask in scopes.items():
            means = scores[:, mask].mean(1)
            calibration[method][scope] = {
                "mean_cosine": float(means.mean()), "permutation_means": means.tolist(),
                "permutation_mean_range95_not_confidence_interval": np.percentile(means, [2.5, 97.5]).tolist(),
                "paired_minus_shuffled": float(scores_by_seed["three_seed_mean"][method][mask].mean() - means.mean()),
            }
    comparisons = {}
    for original in published["seeds"]:
        seed = original["seed"]
        comparisons[str(seed)] = {}
        for scope in scopes:
            old = original["safe_arm"]["scopes"][scope]
            cosine_delta = fidelity[str(seed)][scope]["methods"]["learned_parent"]["mean"] - old["teacher_cosine"]
            auc_delta = attacks[f"learned_parent_seed{seed}"][scope]["metrics"]["strongest_orientation_safe_auc"] - old["attacks"]["strongest_orientation_safe_auc"]
            comparisons[str(seed)][scope] = {
                "teacher_cosine_delta": cosine_delta, "auc_delta": auc_delta,
                "cosine_within_declared_tolerance": abs(cosine_delta) <= cfg["published_comparison_tolerance"]["mean_teacher_cosine_absolute"],
                "auc_within_declared_tolerance": abs(auc_delta) <= cfg["published_comparison_tolerance"]["strongest_auc_absolute"],
            }
    timing = {}
    for method in cfg["methods"]:
        if method in BASELINES:
            ms = [inference_manifest["timings"][method]["median_ms_per_document"]]
        else:
            ms = [inference_manifest["timings"][f"{method}_seed{seed}"]["median_ms_per_document"] for seed in cfg["seeds"]]
        timing[method] = {"mean_checkpoint_median_ms_per_document": float(np.mean(ms)), "checkpoint_medians": ms}
    result = {
        "study": cfg["study"], "analysis_role": cfg["analysis_role"], "configuration": cfg,
        "documents": len(groups), "population_order_sha256": stable_hash(groups.tolist()),
        "inference_manifest_sha256": sha256(output / "INFERENCE_MANIFEST.json"),
        "document_scores_sha256": sha256(output / "DOCUMENT_SCORES.json"),
        "source_sha256": replay_source_hashes() if replay else source_hashes(), "fidelity": fidelity,
        "detectability": aggregate_attacks, "timing": timing,
        "calibration": {"protocol": cfg["calibration"], "permutation_order_sha256": stable_hash(permutations.tolist()), "methods": calibration},
        "published_learned_comparison": comparisons,
        "bootstrap_interpretation": "document-conditional; fixed checkpoints, fitted classifiers, split and selected families; no classifier refit or family reselection in bootstrap",
        "original_results_sha256": require_hash(ROOT / cfg["published_results"], PUBLISHED_HASH),
        "attack_result_sha256": {str(path.relative_to(output)): sha256(path) for path in sorted((output / "attacks").glob("*.json"))},
        "completed_unix": time.time(), "merger_updates": 0,
    }
    if replay:
        result.update({
            "execution_mode": REPLAY_ID,
            "input_view_sha256": inference_manifest["input_view_sha256"],
            "historical_baseline_comparison": compare_archived_baselines(
                cfg, groups, labels, payload["active_children"], raw_scores, fidelity, aggregate_attacks),
            "timing_interpretation": "current hardware only; archived GPU timing is not reproduced or compared",
        })
    atomic_json(output / "RESULTS.json", result)
    _render_report(config_path)


def _render_report(config_path):
    cfg, _, _ = load_protocol(config_path)
    output = ROOT / cfg["output_dir"]
    result = json.loads((output / "RESULTS.json").read_text())
    inference = json.loads((output / "INFERENCE_MANIFEST.json").read_text())
    replay = cfg.get("execution_mode") == REPLAY_ID
    means = result["fidelity"]["three_seed_mean"]["pooled"]["methods"]
    introduction = ("Additive post-review analysis of the same 4,000 documents using a read-only view of cached original official FP32 leaves and teacher targets. This is not the unavailable regenerated-safe baseline cache; no merger retraining or teacher encoding was performed." if replay else
                    "Additive post-review analysis of the previously evaluated 4,000 documents. All methods use the same regenerated FP32 leaves and direct teacher targets; no merger retraining.")
    lines = ["# Matched safe-arm baselines", "", introduction, "",
             "| Representation | Teacher cosine | Delta vs weighted | Delta vs unweighted | Source AUC [95% CI] | ms/document |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    tex = [r"% Generated by scripts/evaluate_safe_baselines.py report; do not edit numbers.",
           r"\begin{table}[H]", (r"\caption{Matched 4,000-prefix comparison on cached official safe tensors, not the historical regenerated-safe inputs. Learned rows average three checkpoints. Timings are from current hardware, not reproduced GPU timings.}" if replay else r"\caption{Matched teacher-comparable prefix comparison on regenerated inputs for the same 4,000 documents. Learned rows average the three final checkpoints; deterministic baselines are evaluated once. $\Delta_w$ and $\Delta_u$ are paired mean teacher-cosine differences against weighted and unweighted pooling. Timings measure device-resident composition only; reference extraction uses the complete forward pass, not an optimized reference-only implementation.}"), r"\label{tab:safe-baselines}", r"\centering\small", r"\begin{tabularx}{\linewidth}{Yrrrrr}", r"\toprule",
           r"Representation & Teacher cosine & $\Delta_w$ & $\Delta_u$ & $A_{\mathrm{safe}}$ & ms/doc \\", r"\midrule"]
    for method in cfg["methods"]:
        cosine = means[method]["mean"]
        dw, du = cosine - means["token_weighted_mean"]["mean"], cosine - means["unweighted_mean"]["mean"]
        attack = result["detectability"][method]["pooled"]
        ms = result["timing"][method]["mean_checkpoint_median_ms_per_document"]
        lines.append(f"| {LABELS[method]} | {cosine:.6f} | {dw:+.6f} | {du:+.6f} | {attack['mean_auc']:.6f} [{attack['ci95'][0]:.6f}, {attack['ci95'][1]:.6f}] | {ms:.6f} |")
        tex.append(f"{LABELS[method]} & {cosine:.4f} & {dw:+.6f} & {du:+.6f} & {attack['mean_auc']:.4f} & {ms:.4f} " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabularx}",
                r"\end{table}"])
    lines.extend(["", "## Paired teacher-cosine contrasts", "", "| Contrast | Mean difference | Paired 95% CI | Fraction positive |", "| --- | ---: | --- | ---: |"])
    for key, summary in result["fidelity"]["three_seed_mean"]["pooled"]["contrasts"].items():
        lines.append(f"| {key} | {summary['mean']:+.9f} | [{summary['ci95'][0]:+.9f}, {summary['ci95'][1]:+.9f}] | {summary['fraction_positive']:.4f} |")
    lines.extend(["", "## Shuffled-target calibration", "", "Within-corpus derangements, no self-pairings; these are scale diagnostics, not competitive baselines or downstream retrieval results.", ""])
    for method in cfg["methods"]:
        score = result["calibration"]["methods"][method]["pooled"]["mean_cosine"]
        lines.append(f"- {LABELS[method]}: {score:.6f}")
    lines.extend(["", "## Provenance and interpretation", "",
                  f"- Device: {inference['environment']['device']}; GPU: {inference['environment']['gpu']}",
                  f"- Timing: {cfg['timing']['boundary']}; float32; batch size {cfg['inference']['batch_size']}; {cfg['timing']['warmup_passes']} warm-up and {cfg['timing']['measured_passes']} measured passes; TF32 disabled.",
                  "- Reference and parent timing use the same measured full-forward call; reference-only speed is not claimed.",
                  "- No teacher encoding, cache loading, transfers, decoder evaluation, or classifier fitting is included in these inference timings.",
                  ("- Cached original official safe tensors used full-document leaf batches (up to 16); the unavailable historical regenerated-safe cache used safe-only batches (up to four). Its tensor bytes are not claimed recovered. Current-hardware timing is not reproduced or compared to archived GPU timing." if replay else
                   "- Original official results remain unchanged. Regenerated tensor byte identity is not claimed; preparation uses safe-only rather than full-document leaf batches."),
                  "- Confidence intervals condition on fixed checkpoints, fitted classifiers, split, and selected classifier families. They do not establish downstream utility or uncertainty over arbitrary training seeds.",
                  "", "## Comparison with published learned results", "",
                  "Tolerance checks diagnose numerical reproducibility, not scientific acceptance; all differences are retained.", "",
                  "| Seed | Scope | Cosine difference | AUC difference | Within tolerances |", "| --- | --- | ---: | ---: | --- |"])
    for seed, scopes in result["published_learned_comparison"].items():
        for scope, entry in scopes.items():
            passed = entry["cosine_within_declared_tolerance"] and entry["auc_within_declared_tolerance"]
            lines.append(f"| {seed} | {scope} | {entry['teacher_cosine_delta']:+.9f} | {entry['auc_delta']:+.9f} | {passed} |")
    if replay:
        comparison = result["historical_baseline_comparison"]
        lines.extend(["", "## Cached official versus historical regenerated-safe results", "",
                      "Differences are replay minus the archived regenerated-safe baseline; all populations and prefix token hashes match. The unavailable historical tensors are not reconstructed. Per-document score differences for all representations, checkpoints and scopes are in RESULTS.json.", "",
                      "| Representation | Replay mean cosine | Archived mean cosine | Cosine difference | Replay AUC | Archived AUC | AUC difference |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for method in cfg["methods"]:
            cosine = comparison["fidelity"]["three_seed_mean"]["pooled"][method]
            auc = comparison["detectability"][method]["pooled"]
            lines.append(f"| {LABELS[method]} | {cosine['replay_mean']:.9f} | {cosine['archived_mean']:.9f} | {cosine['difference']:+.9f} | {auc['replay_auc']:.9f} | {auc['archived_auc']:.9f} | {auc['difference']:+.9f} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "safe_baselines_table.tex").write_text("\n".join(tex) + "\n")
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "ARTIFACT_HASHES.json"]
    atomic_json(output / "ARTIFACT_HASHES.json", {str(path.relative_to(output)): sha256(path) for path in sorted(files)})
    if replay:
        atomic_json(output / "COMPLETE.json", {
            "execution_mode": REPLAY_ID, "config_sha256": cfg["config_sha256"],
            "inference_manifest_sha256": sha256(output / "INFERENCE_MANIFEST.json"),
            "results_sha256": sha256(output / "RESULTS.json"),
            "artifact_hashes_sha256": sha256(output / "ARTIFACT_HASHES.json"),
        })
    print(f"Report written: {output / 'REPORT.md'}", flush=True)


def render(config_path):
    cfg, _, _ = load_protocol(config_path)
    if cfg.get("execution_mode") == REPLAY_ID:
        load_replay_results(cfg)
        print(f"Verified completed report: {ROOT / cfg['output_dir'] / 'REPORT.md'}", flush=True)
        return
    _render_report(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "evaluate", "report", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage in ("infer", "all"):
            infer(args.config, args.device)
        if args.stage in ("evaluate", "all"):
            evaluate(args.config)
        if args.stage == "report":
            render(args.config)
