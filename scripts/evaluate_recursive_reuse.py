#!/usr/bin/env python3
"""Matched direct-versus-binary-reuse study on the 606 exact-four safe prefixes."""
from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from safe_baselines_protocol import (
    ROOT, REPLAY_ID, atomic_json, atomic_tensor, checkpoint_entries, configure_device,
    environment, load_protocol as load_baseline_protocol, load_safe_cache,
    require_hash, sha256, stable_hash,
)
sys.path.insert(0, str(ROOT / "src"))

from evaluate_safe_baselines import load_replay_results as load_baseline_replay_results, replay_checkpoint_identity
from public_replay import package_file_hash
from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.recursive_reuse import (
    CONTRASTS, binary_reuse_four, population_summary, summarize_tree_scores, validate_four,
)
from recursive_geodesic_composition.safe_baselines import (
    baseline_parent, benchmark, cosine_rows, validate_vectors,
)

CONFIG = ROOT / "configs/recursive_reuse_v1.yaml"
REPLAY_CONFIG = ROOT / "configs/recursive_reuse_replay_v1.yaml"
REPLAY_OUTPUT = "replays/recursive_reuse_v1"
LABELS = {"flat_unweighted": "Flat unweighted pooling", "flat_token_weighted": "Flat token-weighted pooling",
          "direct_four": "Direct four-child merge", "recursive_binary": "Recursive binary tree"}


def load_protocol(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if "execution_mode" in cfg:
        return load_replay_protocol(path, cfg)
    if cfg["study"] != "recursive_reuse_v1" or cfg["output_dir"] != "results/recursive_reuse_v1":
        raise ValueError("wrong study/output namespace")
    if cfg["methods"] != list(LABELS) or cfg["seeds"] != [424242, 271828, 314159]:
        raise ValueError("methods or checkpoints changed")
    if cfg["expected_documents"] != 606 or cfg["exact_children"] != 4 or cfg["bootstrap"]["reps"] != 2000:
        raise ValueError("population/bootstrap changed")
    if any(cfg[key] != 0 for key in ("merger_updates", "teacher_encoding_calls", "classifier_fits")):
        raise ValueError("this experiment is merger inference only")
    if (cfg["direct_tree"], cfg["recursive_tree"]) != ("f4(x1,x2,x3,x4)", "f2(f2(x1,x2),f2(x3,x4))"):
        raise ValueError("tree definitions changed")
    for key in ("baseline_config", "baseline_results", "baseline_inference", "exact_four_config"):
        require_hash(ROOT / cfg[key], cfg[f"{key}_sha256"])
    cfg["config_sha256"] = sha256(path)
    baseline, official, published = load_baseline_protocol(ROOT / cfg["baseline_config"])
    return cfg, baseline, official, published


def load_replay_protocol(path, replay):
    expected = {
        "execution_mode": REPLAY_ID,
        "study": "recursive_reuse_v1",
        "scientific_config": "configs/recursive_reuse_v1.yaml",
        "scientific_config_sha256": "427850a584f11618d6ec7727162085b4a7cc777e99a677d7b6dee255a0a9b113",
        "baseline_replay_config": "configs/safe_baselines_replay_v1.yaml",
        "baseline_replay_config_sha256": "e3d54b3e3e3941b59552e7d6ff2fb4081fbcef19214c31f9fc2b4c9abf010608",
        "output_dir": REPLAY_OUTPUT,
    }
    if replay != expected or Path(path).resolve() != REPLAY_CONFIG.resolve():
        raise ValueError("invalid cached-replay recursive configuration or namespace")
    if package_file_hash(ROOT, expected["scientific_config"]) != expected["scientific_config_sha256"]:
        raise ValueError("historical recursive scientific configuration changed")
    package_file_hash(ROOT, "configs/v1_k4_pooling.yaml")
    require_hash(ROOT / expected["baseline_replay_config"], expected["baseline_replay_config_sha256"])
    cfg, _, _, _ = load_protocol(CONFIG)
    baseline, official, published = load_baseline_protocol(ROOT / expected["baseline_replay_config"])
    historical_inputs = {
        cfg[key]: cfg[f"{key}_sha256"] for key in ("baseline_config", "baseline_results", "baseline_inference")
    }
    cfg.update({
        "execution_mode": REPLAY_ID,
        "replay_config": str(REPLAY_CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(path),
        "scientific_config": expected["scientific_config"],
        "scientific_config_sha256": expected["scientific_config_sha256"],
        "historical_frozen_baseline_inputs": historical_inputs,
        "baseline_config": expected["baseline_replay_config"],
        "baseline_config_sha256": expected["baseline_replay_config_sha256"],
        "baseline_results": f"{baseline['output_dir']}/RESULTS.json",
        "baseline_inference": f"{baseline['output_dir']}/INFERENCE_MANIFEST.json",
        "output_dir": REPLAY_OUTPUT,
    })
    # Historical vector/result pins remain in historical_frozen_baseline_inputs;
    # the new result and vector hashes are frozen only after the fresh run exists.
    del cfg["baseline_results_sha256"], cfg["baseline_inference_sha256"]
    return cfg, baseline, official, published


def implementation_hashes():
    files = ["configs/recursive_reuse_v1.yaml", "scripts/evaluate_recursive_reuse.py",
             "scripts/render_recursive_reuse.py", "scripts/audit_recursive_reuse.py",
             "scripts/safe_baselines_protocol.py",
             "src/recursive_geodesic_composition/recursive_reuse.py",
             "src/recursive_geodesic_composition/safe_baselines.py",
             "src/recursive_geodesic_composition/architecture.py",
             "src/recursive_geodesic_composition/evaluate_kary_validation.py"]
    return {name: sha256(ROOT / name) for name in files if (ROOT / name).exists()}


def replay_implementation_hashes():
    files = ("configs/recursive_reuse_replay_v1.yaml", "configs/safe_baselines_replay_v1.yaml",
             "scripts/evaluate_safe_baselines.py", "scripts/public_replay.py")
    return implementation_hashes() | {name: sha256(ROOT / name) for name in files}


def replay_immutable_inputs(cfg, baseline, reference_manifest):
    historical = (
        cfg["scientific_config"], cfg["exact_four_config"], baseline["published_results"],
        "results/recursive_reuse_v1/RESULTS.json",
        "results/recursive_reuse_v1/DOCUMENT_METADATA.json",
        "results/recursive_reuse_v1/DOCUMENT_SCORES.json",
    )
    immutable = {name: package_file_hash(ROOT, name) for name in historical}
    for name, digest in cfg["historical_frozen_baseline_inputs"].items():
        if package_file_hash(ROOT, name) != digest:
            raise ValueError("historical baseline identity changed")
        immutable[name] = digest
    immutable[cfg["replay_config"]] = cfg["config_sha256"]
    immutable[cfg["baseline_config"]] = cfg["baseline_config_sha256"]
    if baseline["config_sha256"] != cfg["baseline_config_sha256"]:
        raise ValueError("upstream baseline replay configuration changed")
    for name in (cfg["baseline_results"], cfg["baseline_inference"],
                 f"{baseline['output_dir']}/vectors.pt",
                 f"{baseline['output_dir']}/DOCUMENT_METADATA.json",
                 f"{baseline['output_dir']}/INPUT_VIEW.json",
                 f"{baseline['output_dir']}/ARTIFACT_HASHES.json",
                 f"{baseline['output_dir']}/COMPLETE.json"):
        immutable[name] = sha256(ROOT / name)
    if (immutable[f"{baseline['output_dir']}/vectors.pt"] != reference_manifest["vectors_sha256"] or
            immutable[f"{baseline['output_dir']}/DOCUMENT_METADATA.json"] != reference_manifest["metadata_sha256"] or
            immutable[f"{baseline['output_dir']}/INPUT_VIEW.json"] != reference_manifest["input_view_sha256"]):
        raise ValueError("upstream baseline files disagree with their inference manifest")
    return immutable


def select_exact_four(rows, cfg, published):
    reference = yaml.safe_load((ROOT / cfg["exact_four_config"]).read_text())["official_population"]
    indices = [index for index, row in enumerate(rows) if len(row["lengths"]) == 4]
    chosen = [rows[index] for index in indices]
    if len(chosen) != cfg["expected_documents"]:
        raise ValueError("exact-four population count changed")
    for corpus in published["corpus_order"]:
        ids = [row["id"] for row in chosen if row["corpus"] == corpus]
        if (len(ids) != reference["expected_exact_k4_per_corpus"][corpus] or
                stable_hash(ids) != reference["expected_exact_k4_order_sha256"][corpus]):
            raise ValueError(f"exact-four document identity/order changed: {corpus}")
    return chosen, np.asarray(indices)


def characterize_population(metadata, cfg, published):
    settings = cfg["population"]
    summary = population_summary(metadata, settings["length_bin_upper_inclusive"], settings["documented_backbone_context_tokens"])
    # Check the FULL discrete distributions against all original seeds, not just
    # a match of totals, so descriptive long-document claims cannot silently drift.
    for seed in published["seeds"]:
        original = seed["full_document_arm"]["document_distributions"]
        for new, old in (("token_histogram", "token_count"), ("leaf_histogram", "leaf_count"), ("depth_histogram", "tree_depth")):
            if summary[new] != original[old]["histogram"]:
                raise ValueError(f"full-document {old} histogram differs from official records")
        if summary["internal_merges"] != seed["full_document_arm"]["throughput"]["internal_merges"]:
            raise ValueError("full-document merge accounting differs")
    per_corpus = {}
    for corpus in published["corpus_order"]:
        part = population_summary([row for row in metadata if row["corpus"] == corpus], settings["length_bin_upper_inclusive"], settings["documented_backbone_context_tokens"])
        old = published["seeds"][0]["full_document_arm"]["document_distributions"]["by_corpus"][corpus]
        if (part["tokens"] != old["token_count"]["sum"] or part["leaves"] != old["leaf_count"]["sum"] or
                part["depth_histogram"] != old["tree_depth"]["histogram"]):
            raise ValueError(f"full population corpus accounting differs: {corpus}")
        per_corpus[corpus] = part
    return {"pooled": summary, "by_corpus": per_corpus,
            "original_global_histograms_verified_for_all_three_seeds": True,
            "source": ("full_token_count metadata from the hash-verified cached official safe tensors; verified against original full-arm histograms"
                       if cfg.get("execution_mode") == REPLAY_ID else
                       "full_token_count metadata from the regenerated, identity-matched 4000 documents; verified against original full-arm histograms"),
            "no_new_full_document_inference": True}


def vector_health(values):
    values = validate_vectors(values, unit=True)
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    return {"all_finite": True, "unit_norm_check_passed": True,
            "norm_min": float(norms.min()), "norm_max": float(norms.max()), "vectors": len(values)}


def infer(config_path, device_name):
    cfg, baseline, official, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    output = ROOT / cfg["output_dir"]
    replay = cfg.get("execution_mode") == REPLAY_ID
    if output.exists():
        if replay and (output / "INFERENCE_MANIFEST.json").exists():
            load_inference(cfg)
            print("Verified existing recursive replay inference; not recomputing.", flush=True)
            return
        raise FileExistsError("study output is create-only; use analyse/report/audit for saved inference")
    rows, cache_manifest = load_safe_cache(baseline, published)
    if replay:
        upstream, reference_manifest, _ = load_baseline_replay_results(baseline)
        reference = {
            "vectors": {key: torch.from_numpy(value) for key, value in upstream["vectors"].items()},
            "targets": torch.from_numpy(upstream["targets"]),
            "groups": upstream["groups"].tolist(),
        }
        if cache_manifest != json.loads((ROOT / baseline["output_dir"] / "INPUT_VIEW.json").read_text()):
            raise ValueError("recursive replay and baseline used different public input views")
    else:
        reference_manifest = json.loads((ROOT / cfg["baseline_inference"]).read_text())
        reference_path = ROOT / baseline["output_dir"] / "vectors.pt"
        require_hash(reference_path, reference_manifest["vectors_sha256"])
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    cache_manifest_path = ROOT / baseline["cache_dir"] / "CACHE_MANIFEST.json"
    require_hash(cache_manifest_path, reference_manifest["cache_manifest_sha256"])
    if [row["id"] for row in rows] != reference["groups"]:
        raise ValueError("prior inference and cache document identities differ")
    selected, indices = select_exact_four(rows, cfg, published)
    checkpoints = checkpoint_entries(baseline)
    if replay:
        immutable = replay_immutable_inputs(cfg, baseline, reference_manifest)
        sources = replay_implementation_hashes()
    else:
        immutable = {cfg[key]: cfg[f"{key}_sha256"] for key in ("baseline_config", "baseline_results", "baseline_inference", "exact_four_config")}
        immutable[baseline["published_results"]] = baseline["published_results_sha256"]
        sources = implementation_hashes()
    output.mkdir(parents=True)
    runtime = environment(device)
    if replay:
        runtime.pop("git_commit", None)
    freeze = {"configuration": cfg, "source_sha256": sources,
              "immutable_input_sha256": immutable, "environment": runtime,
              "cache_manifest_sha256": sha256(cache_manifest_path), "started_unix": time.time(),
              "analysis_role": cfg["analysis_role"], "not_a_depth_only_ablation": True}
    if replay:
        freeze.update({
            "execution_mode": REPLAY_ID,
            "input_view_sha256": reference_manifest["input_view_sha256"],
            "upstream_baseline": {
                "config_path": cfg["baseline_config"], "config_sha256": cfg["baseline_config_sha256"],
                "inference_manifest_path": cfg["baseline_inference"],
                "inference_manifest_sha256": immutable[cfg["baseline_inference"]],
                "results_path": cfg["baseline_results"], "results_sha256": immutable[cfg["baseline_results"]],
                "vectors_path": f"{baseline['output_dir']}/vectors.pt",
                "vectors_sha256": reference_manifest["vectors_sha256"],
                "completion_sha256": immutable[f"{baseline['output_dir']}/COMPLETE.json"],
            },
            "timing_interpretation": "current hardware only; archived GPU timing is not reproduced or compared",
        })
    atomic_json(output / "ANALYSIS_PROTOCOL.json", freeze)

    children = torch.stack([row["leaves"] for row in selected]).float().to(device)
    counts = torch.stack([row["lengths"] for row in selected]).float().to(device)
    validate_four(children, counts)
    targets = torch.stack([row["teacher_target"] for row in selected])
    if not torch.equal(targets, reference["targets"][indices]):
        raise ValueError("teacher targets differ from improvement 1")
    batch_size = int(cfg["inference"]["batch_size"])
    batches = [(children[start:start + batch_size], counts[start:start + batch_size]) for start in range(0, len(selected), batch_size)]
    vectors, intermediate_vectors, timings, validation = {}, {}, {}, {}
    warmup, repeats = cfg["timing"]["warmup_passes"], cfg["timing"]["measured_passes"]
    with torch.inference_mode():
        for name, method in (("flat_unweighted", "unweighted_mean"), ("flat_token_weighted", "token_weighted_mean")):
            value = torch.cat([baseline_parent(x, n, method).cpu() for x, n in batches])
            delta = float((value - reference["vectors"][method][indices]).abs().max())
            if delta > cfg["reference_vector_max_absolute_tolerance"]:
                raise ValueError("flat pooling does not match the prior exact-four subset")
            vectors[name] = value
            validation[name] = {"max_abs_difference_from_previous_vectors": delta}
            timings[name] = benchmark(lambda x, n: baseline_parent(x, n, method), batches, device, warmup, repeats, len(selected))
        for entry in checkpoints:
            seed = entry["seed"]
            model, checkpoint = build_model(ROOT / entry["path"], device)
            if checkpoint.get("seed") != seed or model.arity != 4 or model.d != 768:
                raise ValueError("wrong checkpoint identity or architecture")
            direct, recursive, intermediate = [], [], []
            for x, n in batches:
                direct.append(model.merge(x, n).cpu())
                root, pair_parents, pair_counts = binary_reuse_four(model, x, n, return_intermediates=True)
                if not torch.equal(pair_counts.sum(1), n.sum(1)):
                    raise ValueError("recursive tree did not conserve token counts")
                recursive.append(root.cpu())
                intermediate.append(pair_parents.cpu())
            direct, recursive, intermediate = torch.cat(direct), torch.cat(recursive), torch.cat(intermediate)
            delta = float((direct - reference["vectors"][f"learned_parent_seed{seed}"][indices]).abs().max())
            if delta > cfg["reference_vector_max_absolute_tolerance"]:
                raise ValueError("direct parent differs from improvement 1 beyond the fixed tolerance")
            vectors[f"direct_four_seed{seed}"] = direct
            vectors[f"recursive_binary_seed{seed}"] = recursive
            intermediate_vectors[str(seed)] = intermediate
            validation[str(seed)] = {"direct_max_abs_difference_from_previous_vectors": delta,
                                     "intermediate_health": vector_health(intermediate.reshape(-1, 768).numpy()),
                                     "exact_count_conservation": True}
            timings[f"direct_four_seed{seed}"] = benchmark(model.merge, batches, device, warmup, repeats, len(selected))
            timings[f"recursive_binary_seed{seed}"] = benchmark(lambda x, n: binary_reuse_four(model, x, n), batches, device, warmup, repeats, len(selected))
            require_hash(ROOT / entry["path"], entry["sha256"])
            print(f"Completed frozen seed {seed}: direct, recursive and matched timing", flush=True)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
    health = {key: vector_health(value.numpy()) for key, value in vectors.items()}
    metadata = [{key: row[key] for key in ("id", "corpus", "source_index", "safe_prefix_token_count", "full_token_count", "safe_prefix_token_ids_sha256")} |
                {"leaf_counts": row["lengths"].to(torch.int64).tolist(),
                 "intermediate_counts": [int(row["lengths"][:2].sum()), int(row["lengths"][2:].sum())]} for row in selected]
    atomic_json(output / "DOCUMENT_METADATA.json", metadata)
    payload = {"vectors": vectors, "intermediate_vectors": intermediate_vectors,
               "targets": targets, "groups": [row["id"] for row in selected],
               "labels": [row["corpus"] for row in selected], "counts": counts.cpu(),
               "intermediate_counts": counts.reshape(len(selected), 2, 2).sum(2).cpu()}
    atomic_tensor(output / "vectors.pt", payload)
    population = characterize_population(rows, cfg, published)
    atomic_json(output / "FULL_POPULATION.json", population)
    for path, digest in immutable.items():
        require_hash(ROOT / path, digest)
    manifest = freeze | {
        "checkpoints": [replay_checkpoint_identity(entry) for entry in checkpoints] if replay else checkpoints,
        "documents": len(selected), "population_order_sha256": stable_hash(payload["groups"]),
        "vectors_sha256": sha256(output / "vectors.pt"), "metadata_sha256": sha256(output / "DOCUMENT_METADATA.json"),
        "full_population_sha256": sha256(output / "FULL_POPULATION.json"), "health": health,
        "reference_checks": validation, "timings": timings, "completed_unix": time.time(),
        "direct_tree": {"depth": 1, "merges_per_document": 1, "active_arities": [4]},
        "recursive_tree": {"depth": 2, "merges_per_document": 3, "active_arities": [2, 2, 2]},
        "new_teacher_encoding_calls": 0, "merger_updates": 0, "classifier_fits": 0,
    }
    if replay:
        for path, digest in sources.items():
            require_hash(ROOT / path, digest)
        manifest["analysis_protocol_sha256"] = sha256(output / "ANALYSIS_PROTOCOL.json")
    atomic_json(output / "INFERENCE_MANIFEST.json", manifest)


def load_inference(cfg):
    output = ROOT / cfg["output_dir"]
    manifest = json.loads((output / "INFERENCE_MANIFEST.json").read_text())
    if manifest["configuration"]["config_sha256"] != cfg["config_sha256"]:
        raise ValueError("configuration changed after inference")
    for path, digest in manifest["source_sha256"].items():
        require_hash(ROOT / path, digest)
    for file, field in (("vectors.pt", "vectors_sha256"), ("DOCUMENT_METADATA.json", "metadata_sha256"), ("FULL_POPULATION.json", "full_population_sha256")):
        require_hash(output / file, manifest[field])
    payload = torch.load(output / "vectors.pt", map_location="cpu", weights_only=True)
    if stable_hash(payload["groups"]) != manifest["population_order_sha256"]:
        raise ValueError("vector document order changed")
    if cfg.get("execution_mode") == REPLAY_ID:
        validate_replay_inference(cfg, payload, manifest)
    for key, value in payload["vectors"].items():
        validate_vectors(value.numpy(), key, unit=True)
    return payload, manifest


def validate_replay_inference(cfg, payload, manifest):
    output = ROOT / cfg["output_dir"]
    if (manifest.get("execution_mode") != REPLAY_ID or manifest.get("configuration") != cfg or
            manifest.get("documents") != 606 or manifest.get("new_teacher_encoding_calls") != 0 or
            manifest.get("merger_updates") != 0 or manifest.get("classifier_fits") != 0 or
            manifest.get("not_a_depth_only_ablation") is not True):
        raise ValueError("incomplete or contradictory recursive replay inference")
    if manifest["source_sha256"] != replay_implementation_hashes():
        raise ValueError("recursive replay implementation changed")
    require_hash(output / "ANALYSIS_PROTOCOL.json", manifest["analysis_protocol_sha256"])
    protocol = json.loads((output / "ANALYSIS_PROTOCOL.json").read_text())
    if any(manifest.get(key) != value for key, value in protocol.items()):
        raise ValueError("recursive replay freeze disagrees with its inference manifest")
    baseline, _, _ = load_baseline_protocol(ROOT / cfg["baseline_config"])
    upstream, reference_manifest, _ = load_baseline_replay_results(baseline)
    immutable = replay_immutable_inputs(cfg, baseline, reference_manifest)
    if manifest["immutable_input_sha256"] != immutable:
        raise ValueError("recursive replay upstream baseline or historical evidence changed")
    for path, digest in immutable.items():
        require_hash(ROOT / path, digest)
    expected_upstream = {
        "config_path": cfg["baseline_config"], "config_sha256": cfg["baseline_config_sha256"],
        "inference_manifest_path": cfg["baseline_inference"],
        "inference_manifest_sha256": immutable[cfg["baseline_inference"]],
        "results_path": cfg["baseline_results"], "results_sha256": immutable[cfg["baseline_results"]],
        "vectors_path": f"{baseline['output_dir']}/vectors.pt",
        "vectors_sha256": reference_manifest["vectors_sha256"],
        "completion_sha256": immutable[f"{baseline['output_dir']}/COMPLETE.json"],
    }
    if (manifest["upstream_baseline"] != expected_upstream or
            manifest["input_view_sha256"] != reference_manifest["input_view_sha256"] or
            manifest["cache_manifest_sha256"] != reference_manifest["cache_manifest_sha256"]):
        raise ValueError("recursive replay is not bound to the completed current baseline")
    if manifest["checkpoints"] != [replay_checkpoint_identity(entry) for entry in checkpoint_entries(baseline)]:
        raise ValueError("recursive replay checkpoint identities changed")
    metadata = json.loads((output / "DOCUMENT_METADATA.json").read_text())
    archived_metadata = json.loads((ROOT / "results/recursive_reuse_v1/DOCUMENT_METADATA.json").read_text())
    if metadata != archived_metadata or len(metadata) != 606:
        raise ValueError("recursive replay exact-four identities, counts, or token hashes changed")
    indices = np.flatnonzero(upstream["active_children"] == 4)
    if (len(indices) != 606 or payload["groups"] != upstream["groups"][indices].tolist() or
            payload["labels"] != upstream["labels"][indices].tolist() or
            payload["groups"] != [row["id"] for row in metadata]):
        raise ValueError("recursive replay is not the paired exact-four baseline population")
    counts = payload["counts"]
    expected_counts = torch.tensor([row["leaf_counts"] for row in metadata], dtype=torch.float32)
    if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.float32 or
            counts.shape != (606, 4) or not torch.equal(counts, expected_counts) or
            not torch.equal(payload["intermediate_counts"], counts.reshape(606, 2, 2).sum(2))):
        raise ValueError("recursive replay did not preserve exact token counts")
    target = payload["targets"]
    if (not isinstance(target, torch.Tensor) or target.dtype != torch.float32 or
            target.shape != (606, 768) or
            not torch.equal(target, torch.from_numpy(upstream["targets"][indices]))):
        raise ValueError("recursive teacher targets differ from the current baseline")
    validate_vectors(target.numpy(), "recursive teacher targets", unit=True)
    keys = {"flat_unweighted", "flat_token_weighted"} | {
        f"{method}_seed{seed}" for method in ("direct_four", "recursive_binary") for seed in cfg["seeds"]}
    if set(payload["vectors"]) != keys or set(manifest["timings"]) != keys:
        raise ValueError("recursive replay omitted a method, checkpoint, or timing")
    if set(payload["intermediate_vectors"]) != {str(seed) for seed in cfg["seeds"]}:
        raise ValueError("recursive replay omitted checkpoint intermediate vectors")
    for key, values in payload["vectors"].items():
        if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.shape != (606, 768):
            raise ValueError(f"{key}: recursive replay vector shape or dtype changed")
        validate_vectors(values.numpy(), key, unit=True)
        duration = manifest["timings"][key]["median_ms_per_document"]
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"{key}: invalid current-hardware timing")
    for seed in cfg["seeds"]:
        values = payload["intermediate_vectors"][str(seed)]
        if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.shape != (606, 2, 768):
            raise ValueError("recursive intermediate vectors have the wrong shape or dtype")
        validate_vectors(values.reshape(-1, 768).numpy(), "recursive intermediate vectors", unit=True)
    references = {
        "flat_unweighted": "unweighted_mean", "flat_token_weighted": "token_weighted_mean",
        **{f"direct_four_seed{seed}": f"learned_parent_seed{seed}" for seed in cfg["seeds"]},
    }
    for key, upstream_key in references.items():
        delta = float((payload["vectors"][key] - torch.from_numpy(upstream["vectors"][upstream_key][indices])).abs().max())
        if delta > cfg["reference_vector_max_absolute_tolerance"]:
            raise ValueError(f"{key}: vector differs from the current replay baseline")
    population = json.loads((output / "FULL_POPULATION.json").read_text())
    if population["pooled"]["documents"] != 4000 or population["original_global_histograms_verified_for_all_three_seeds"] is not True:
        raise ValueError("recursive replay full-population evidence is incomplete")


def compare_archived_recursive(cfg, payload, scores_by_seed, summaries, scopes):
    results_path = "results/recursive_reuse_v1/RESULTS.json"
    documents_path = "results/recursive_reuse_v1/DOCUMENT_SCORES.json"
    results_hash = package_file_hash(ROOT, results_path)
    documents_hash = package_file_hash(ROOT, documents_path)
    archived = json.loads((ROOT / results_path).read_text())
    documents = json.loads((ROOT / documents_path).read_text())
    if (archived["configuration"]["config_sha256"] != cfg["scientific_config_sha256"] or
            archived["documents"] != 606 or len(documents) != 606 or
            archived["population_order_sha256"] != stable_hash(payload["groups"])):
        raise ValueError("archived recursive comparison population or scientific protocol changed")
    for index, row in enumerate(documents):
        if row["id"] != payload["groups"][index] or row["corpus"] != payload["labels"][index]:
            raise ValueError("archived recursive scores are not paired to the replay documents")
    comparison = {}
    for seed, cells in summaries.items():
        comparison[seed] = {}
        for scope, new in cells.items():
            old = archived["summaries"][seed][scope]
            comparison[seed][scope] = {}
            for block in ("metrics", "contrasts"):
                comparison[seed][scope][block] = {}
                for name, value in new[block].items():
                    previous = old[block][name]
                    comparison[seed][scope][block][name] = {
                        "replay_mean": value["mean"], "archived_mean": previous["mean"],
                        "difference": value["mean"] - previous["mean"],
                        "replay_ci95": value["ci95"], "archived_ci95": previous["ci95"],
                    }
            new_loss, old_loss = new["degradation_direct_minus_recursive"], old["degradation_direct_minus_recursive"]
            comparison[seed][scope]["fraction_degraded"] = {
                "replay": new_loss["fraction_degraded"], "archived": old_loss["fraction_degraded"],
                "difference": new_loss["fraction_degraded"] - old_loss["fraction_degraded"],
            }
    document_differences = {}
    for seed, columns in scores_by_seed.items():
        document_differences[seed] = {}
        for name, values in columns.items():
            old = np.asarray([row["scores"][seed][name] for row in documents], dtype=np.float64)
            if not np.isfinite(old).all():
                raise ValueError(f"{seed}/{name}: nonfinite archived recursive scores")
            document_differences[seed][name] = {}
            for scope, mask in scopes.items():
                delta = values[mask] - old[mask]
                document_differences[seed][name][scope] = {
                    "documents": int(mask.sum()), "replay_mean": float(values[mask].mean()),
                    "archived_mean": float(old[mask].mean()),
                    "mean_signed_difference": float(delta.mean()),
                    "mean_absolute_difference": float(np.abs(delta).mean()),
                    "maximum_absolute_difference": float(np.abs(delta).max()),
                }
    return {
        "comparison": "cached original official safe tensors versus historical regenerated-safe recursive results",
        "archived_results_path": results_path, "archived_results_sha256": results_hash,
        "archived_document_scores_path": documents_path, "archived_document_scores_sha256": documents_hash,
        "paired_document_identities_verified": True, "summaries": comparison,
        "document_score_differences": document_differences, "timings_compared": False,
    }


def load_replay_results(cfg):
    if cfg.get("execution_mode") != REPLAY_ID:
        raise ValueError("completed recursive replay loader requires its replay configuration")
    output = ROOT / cfg["output_dir"]
    marker = json.loads((output / "COMPLETE.json").read_text())
    if marker.get("execution_mode") != REPLAY_ID or marker.get("config_sha256") != cfg["config_sha256"]:
        raise ValueError("recursive replay completion identity changed")
    require_hash(output / "ARTIFACT_HASHES.json", marker["artifact_hashes_sha256"])
    require_hash(output / "RESULTS.json", marker["results_sha256"])
    require_hash(output / "INFERENCE_MANIFEST.json", marker["inference_manifest_sha256"])
    payload, manifest = load_inference(cfg)
    result = json.loads((output / "RESULTS.json").read_text())
    if (result.get("execution_mode") != REPLAY_ID or result.get("configuration") != cfg or
            result.get("documents") != 606 or result.get("checkpoint_order") != cfg["seeds"] or
            result.get("source_sha256") != manifest["source_sha256"] or
            result.get("immutable_input_sha256") != manifest["immutable_input_sha256"] or
            result.get("upstream_baseline") != manifest["upstream_baseline"] or
            result.get("inference_manifest_sha256") != marker["inference_manifest_sha256"] or
            result.get("population_order_sha256") != manifest["population_order_sha256"]):
        raise ValueError("recursive replay results contradict the completed inference")
    require_hash(output / "DOCUMENT_SCORES.json", result["document_scores_sha256"])
    require_hash(output / "FULL_POPULATION.json", result["full_population_sha256"])
    comparison = result["historical_recursive_comparison"]
    for path_key, hash_key in (("archived_results_path", "archived_results_sha256"),
                               ("archived_document_scores_path", "archived_document_scores_sha256")):
        if package_file_hash(ROOT, comparison[path_key]) != comparison[hash_key]:
            raise ValueError("archived recursive comparison identity changed")
    if comparison["timings_compared"] is not False:
        raise ValueError("archived hardware timing must not be treated as reproduced")
    expected = {
        "ANALYSIS_PROTOCOL.json", "INFERENCE_MANIFEST.json", "vectors.pt",
        "DOCUMENT_METADATA.json", "FULL_POPULATION.json", "DOCUMENT_SCORES.json",
        "RESULTS.json", "REPORT.md",
    }
    paths = list(output.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("completed recursive replay contains a symbolic link")
    actual = {str(path.relative_to(output)) for path in paths if path.is_file()}
    if actual != expected | {"ARTIFACT_HASHES.json", "COMPLETE.json"}:
        raise ValueError("completed recursive replay has missing or unexpected artifacts")
    hashes = json.loads((output / "ARTIFACT_HASHES.json").read_text())
    if set(hashes) != expected:
        raise ValueError("recursive replay artifact inventory changed")
    for name, digest in hashes.items():
        require_hash(output / name, digest)
    return payload, manifest, result


def _render_replay_report(cfg, result, manifest):
    output = ROOT / cfg["output_dir"]
    summary = result["summaries"]["three_seed_mean"]["pooled"]
    comparison = result["historical_recursive_comparison"]["summaries"]["three_seed_mean"]["pooled"]
    upstream = manifest["upstream_baseline"]
    lines = [
        "# Anonymous cached recursive-reuse replay", "",
        "All 606 exact-four prefixes and all three frozen checkpoints; 2,000 shared-document bootstrap resamples. The same scientific binary-reuse, direct-merge and flat-pooling implementations are used without retraining, teacher encoding or classifier fitting.", "",
        "Inputs are the unchanged FP32 tensors from the published original official safe cache, not the unavailable historical regenerated-safe cache. Population, document order, token hashes and exact token masses are checked against archived metadata.", "",
        "Direct: `f4(x1,x2,x3,x4)`. Recursive: `f2(f2(x1,x2),f2(x3,x4))`, with the same K4 checkpoint, two active leading slots and exact summed counts. This changes active arity and tree structure together; it is not a depth-only ablation.", "",
        "| Representation | Replay teacher cosine | Archived regenerated-safe cosine | Difference | Current ms/document |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in cfg["methods"]:
        value = comparison["metrics"][method]
        cost = result["timing"][method]["mean_checkpoint_median_ms_per_document"]
        lines.append(f"| {LABELS[method]} | {value['replay_mean']:.9f} | {value['archived_mean']:.9f} | {value['difference']:+.9f} | {cost:.6f} |")
    lines.extend(["", "## Shared-document paired contrasts", "",
                  "| Contrast | Replay mean | Replay 95% CI | Archived mean | Difference |",
                  "| --- | ---: | --- | ---: | ---: |"])
    for name, entry in comparison["contrasts"].items():
        interval = entry["replay_ci95"]
        lines.append(f"| {name} | {entry['replay_mean']:+.9f} | [{interval[0]:+.9f}, {interval[1]:+.9f}] | {entry['archived_mean']:+.9f} | {entry['difference']:+.9f} |")
    loss = summary["degradation_direct_minus_recursive"]
    lines.extend([
        "", f"Fraction degraded: {loss['fraction_degraded']:.9f}; fraction improved: {loss['fraction_improved']:.9f}.",
        "All checkpoint/scope metric and contrast differences and per-document mean/max absolute score differences are retained in RESULTS.json. Numerical differences are measured, not used to tune or normalize replay outputs.", "",
        "## Fresh upstream contract", "",
        f"- Baseline replay config: `{upstream['config_path']}`; SHA-256 `{upstream['config_sha256']}`.",
        f"- Baseline inference manifest: `{upstream['inference_manifest_path']}`; SHA-256 `{upstream['inference_manifest_sha256']}`.",
        f"- Fresh vectors: `{upstream['vectors_path']}`; SHA-256 `{upstream['vectors_sha256']}`.",
        f"- Fresh baseline results: `{upstream['results_path']}`; SHA-256 `{upstream['results_sha256']}`.",
        "Historical missing-vector/config pins remain evidence; none are rebound to these new artifacts.",
        "", "## Timing and interpretation", "",
        f"Device: `{manifest['environment']['device']}`; GPU: `{manifest['environment']['gpu']}`. FP32, TF32 disabled, batch size 64, three warm-up and 20 measured passes.",
        "These are actual current-hardware composition timings. Archived GPU timing is neither reproduced nor compared. Timing excludes cache I/O, transfers, teacher encoding and classifier fitting.",
        "Intervals condition on fixed checkpoints and resample documents jointly across methods and checkpoints; they do not estimate uncertainty over arbitrary training seeds.",
        "Full 4,000-document population accounting is in FULL_POPULATION.json, with full-token/leaf/depth histograms verified against all three original official seeds. No unrestricted full-document inference or new full-root teacher targets are claimed.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(output / "ARTIFACT_HASHES.json", {
        str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in ("ARTIFACT_HASHES.json", "COMPLETE.json")
    })
    atomic_json(output / "COMPLETE.json", {
        "execution_mode": REPLAY_ID, "config_sha256": cfg["config_sha256"],
        "inference_manifest_sha256": sha256(output / "INFERENCE_MANIFEST.json"),
        "results_sha256": sha256(output / "RESULTS.json"),
        "artifact_hashes_sha256": sha256(output / "ARTIFACT_HASHES.json"),
    })
    print(f"Completed recursive replay report: {output / 'REPORT.md'}", flush=True)


def render(config_path):
    cfg, _, _, _ = load_protocol(config_path)
    if cfg.get("execution_mode") != REPLAY_ID:
        raise ValueError("historical reports use scripts/render_recursive_reuse.py")
    load_replay_results(cfg)
    print(f"Verified completed report: {ROOT / cfg['output_dir'] / 'REPORT.md'}", flush=True)


def analyse(config_path):
    cfg, baseline, official, published = load_protocol(config_path)
    output = ROOT / cfg["output_dir"]
    replay = cfg.get("execution_mode") == REPLAY_ID
    if (output / "RESULTS.json").exists():
        if replay:
            load_replay_results(cfg)
            print("Verified completed recursive replay; not recomputing.", flush=True)
            return
        raise FileExistsError("results already exist; use report or audit")
    payload, manifest = load_inference(cfg)
    if replay:
        expected = {"ANALYSIS_PROTOCOL.json", "INFERENCE_MANIFEST.json", "vectors.pt",
                    "DOCUMENT_METADATA.json", "FULL_POPULATION.json"}
        if {path.name for path in output.iterdir()} != expected:
            raise FileExistsError("incomplete or contradictory recursive analysis exists; use a fresh output namespace")
    targets = payload["targets"].numpy()
    raw_scores = {key: cosine_rows(value.numpy(), targets) for key, value in payload["vectors"].items()}
    scores_by_seed = {}
    for seed in cfg["seeds"]:
        scores = {method: raw_scores[f"{method}_seed{seed}"] if method in ("direct_four", "recursive_binary") else raw_scores[method]
                  for method in cfg["methods"]}
        scores["direct_recursive_output_cosine"] = cosine_rows(payload["vectors"][f"direct_four_seed{seed}"].numpy(), payload["vectors"][f"recursive_binary_seed{seed}"].numpy())
        scores_by_seed[str(seed)] = scores
    scores_by_seed["three_seed_mean"] = {name: np.mean([scores_by_seed[str(seed)][name] for seed in cfg["seeds"]], axis=0)
                                         for name in scores_by_seed[str(cfg["seeds"][0])]}
    labels = np.asarray(payload["labels"])
    scopes = {"pooled": np.ones(len(labels), dtype=bool)} | {corpus: labels == corpus for corpus in published["corpus_order"]}
    summaries = {seed: {scope: summarize_tree_scores(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + index, cfg["loss_thresholds"])
                       for index, (scope, mask) in enumerate(scopes.items())}
                 for seed, scores in scores_by_seed.items()}
    documents = []
    for index, group in enumerate(payload["groups"]):
        documents.append({"id": group, "corpus": payload["labels"][index],
                          "scores": {seed: {name: float(value[index]) for name, value in scores.items()} for seed, scores in scores_by_seed.items()},
                          "contrasts": {seed: {name: float(scores[a][index] - scores[b][index]) for name, (a, b) in CONTRASTS.items()} for seed, scores in scores_by_seed.items()},
                          "degradation_direct_minus_recursive": {seed: float(scores["direct_four"][index] - scores["recursive_binary"][index]) for seed, scores in scores_by_seed.items()}})
    atomic_json(output / "DOCUMENT_SCORES.json", documents)
    timing = {}
    for method in cfg["methods"]:
        keys = [f"{method}_seed{seed}" for seed in cfg["seeds"]] if method in ("direct_four", "recursive_binary") else [method]
        medians = [manifest["timings"][key]["median_ms_per_document"] for key in keys]
        timing[method] = {"mean_checkpoint_median_ms_per_document": float(np.mean(medians)), "checkpoint_medians": medians}
    result = {
        "study": cfg["study"], "configuration": cfg, "documents": len(documents),
        "scope_order": list(scopes), "checkpoint_order": cfg["seeds"], "summaries": summaries,
        "timing": timing, "not_a_depth_only_ablation": True,
        "bootstrap_interpretation": "shared document resamples; average checkpoint scores within each document; fixed checkpoints; no training-seed population inference",
        "inference_manifest_sha256": sha256(output / "INFERENCE_MANIFEST.json"),
        "document_scores_sha256": sha256(output / "DOCUMENT_SCORES.json"),
        "full_population_sha256": sha256(output / "FULL_POPULATION.json"),
        "population_order_sha256": stable_hash(payload["groups"]),
        "source_sha256": replay_implementation_hashes() if replay else implementation_hashes(),
        "immutable_input_sha256": manifest["immutable_input_sha256"], "completed_unix": time.time(),
    }
    if replay:
        result.update({
            "execution_mode": REPLAY_ID, "upstream_baseline": manifest["upstream_baseline"],
            "historical_recursive_comparison": compare_archived_recursive(
                cfg, payload, scores_by_seed, summaries, scopes),
            "timing_interpretation": "current hardware only; archived GPU timing is not reproduced or compared",
        })
    for path, digest in result["immutable_input_sha256"].items():
        require_hash(ROOT / path, digest)
    atomic_json(output / "RESULTS.json", result)
    if replay:
        _render_replay_report(cfg, result, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "analyse", "report", "all"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage in ("infer", "all"):
        infer(args.config, args.device)
    if args.stage in ("analyse", "all"):
        analyse(args.config)
    if args.stage == "report":
        render(args.config)
