#!/usr/bin/env python3
"""Matched direct-versus-binary-reuse study on the 606 exact-four safe prefixes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from safe_baselines_protocol import (
    ROOT, atomic_json, atomic_tensor, checkpoint_entries, configure_device,
    environment, load_protocol as load_baseline_protocol, load_safe_cache,
    require_hash, sha256, stable_hash,
)
from recursive_geodesic_composition.evaluate_kary_validation import build_model
from recursive_geodesic_composition.recursive_reuse import (
    CONTRASTS, binary_reuse_four, population_summary, summarize_tree_scores, validate_four,
)
from recursive_geodesic_composition.safe_baselines import (
    baseline_parent, benchmark, cosine_rows, validate_vectors,
)

CONFIG = ROOT / "configs/recursive_reuse_v1.yaml"
LABELS = {"flat_unweighted": "Flat unweighted pooling", "flat_token_weighted": "Flat token-weighted pooling",
          "direct_four": "Direct four-child merge", "recursive_binary": "Recursive binary tree"}


def load_protocol(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
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


def implementation_hashes():
    files = ["configs/recursive_reuse_v1.yaml", "scripts/evaluate_recursive_reuse.py",
             "scripts/render_recursive_reuse.py", "scripts/audit_recursive_reuse.py",
             "scripts/safe_baselines_protocol.py",
             "src/recursive_geodesic_composition/recursive_reuse.py",
             "src/recursive_geodesic_composition/safe_baselines.py",
             "src/recursive_geodesic_composition/architecture.py",
             "src/recursive_geodesic_composition/evaluate_kary_validation.py"]
    return {name: sha256(ROOT / name) for name in files if (ROOT / name).exists()}


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
            "source": "full_token_count metadata from the regenerated, identity-matched 4000 documents; verified against original full-arm histograms",
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
    if output.exists():
        raise FileExistsError("study output is create-only; use analyse/report/audit for saved inference")
    rows, cache_manifest = load_safe_cache(baseline, published)
    reference_manifest = json.loads((ROOT / cfg["baseline_inference"]).read_text())
    cache_manifest_path = ROOT / baseline["cache_dir"] / "CACHE_MANIFEST.json"
    require_hash(cache_manifest_path, reference_manifest["cache_manifest_sha256"])
    reference_path = ROOT / baseline["output_dir"] / "vectors.pt"
    require_hash(reference_path, reference_manifest["vectors_sha256"])
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if [row["id"] for row in rows] != reference["groups"]:
        raise ValueError("prior inference and cache document identities differ")
    selected, indices = select_exact_four(rows, cfg, published)
    checkpoints = checkpoint_entries(baseline)
    output.mkdir(parents=True)
    immutable = {cfg[key]: cfg[f"{key}_sha256"] for key in ("baseline_config", "baseline_results", "baseline_inference", "exact_four_config")}
    immutable[baseline["published_results"]] = baseline["published_results_sha256"]
    freeze = {"configuration": cfg, "source_sha256": implementation_hashes(),
              "immutable_input_sha256": immutable, "environment": environment(device),
              "cache_manifest_sha256": sha256(cache_manifest_path), "started_unix": time.time(),
              "analysis_role": cfg["analysis_role"], "not_a_depth_only_ablation": True}
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
        "checkpoints": checkpoints, "documents": len(selected), "population_order_sha256": stable_hash(payload["groups"]),
        "vectors_sha256": sha256(output / "vectors.pt"), "metadata_sha256": sha256(output / "DOCUMENT_METADATA.json"),
        "full_population_sha256": sha256(output / "FULL_POPULATION.json"), "health": health,
        "reference_checks": validation, "timings": timings, "completed_unix": time.time(),
        "direct_tree": {"depth": 1, "merges_per_document": 1, "active_arities": [4]},
        "recursive_tree": {"depth": 2, "merges_per_document": 3, "active_arities": [2, 2, 2]},
        "new_teacher_encoding_calls": 0, "merger_updates": 0, "classifier_fits": 0,
    }
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
    for key, value in payload["vectors"].items():
        validate_vectors(value.numpy(), key, unit=True)
    return payload, manifest


def analyse(config_path):
    cfg, baseline, official, published = load_protocol(config_path)
    output = ROOT / cfg["output_dir"]
    if (output / "RESULTS.json").exists():
        raise FileExistsError("results already exist; use report or audit")
    payload, manifest = load_inference(cfg)
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
        "population_order_sha256": stable_hash(payload["groups"]), "source_sha256": implementation_hashes(),
        "immutable_input_sha256": manifest["immutable_input_sha256"], "completed_unix": time.time(),
    }
    for path, digest in result["immutable_input_sha256"].items():
        require_hash(ROOT / path, digest)
    atomic_json(output / "RESULTS.json", result)


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
