#!/usr/bin/env python3
"""Additive post-review safe-arm baselines; never invoke the official run entry point."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

from safe_baselines_protocol import (
    DEFAULT_CONFIG, ROOT, PUBLISHED_HASH, atomic_json, atomic_tensor, checkpoint_entries,
    configure_device, environment, load_protocol, load_safe_cache, require_hash,
    sha256, source_hashes, stable_hash,
)
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
    rows, cache_manifest = load_safe_cache(cfg, published)
    entries = checkpoint_entries(cfg)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "ANALYSIS_PROTOCOL.json", {
        "configuration": cfg, "source_sha256": source_hashes(),
        "environment": environment(device), "started_unix": time.time(),
        "analysis_role": cfg["analysis_role"], "original_official_results_unchanged": True,
        "cache_manifest_sha256": sha256(ROOT / cfg["cache_dir"] / "CACHE_MANIFEST.json"),
    })
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
            model_provenance.append(entry)
            print(f"Inferred and timed checkpoint {seed}", flush=True)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
    metadata = [{key: row[key] for key in (
        "id", "corpus", "source_index", "safe_prefix_token_count", "full_token_count",
        "safe_prefix_token_ids_sha256")} | {"active_children": len(row["lengths"])} for row in rows]
    atomic_json(output / "DOCUMENT_METADATA.json", metadata)
    payload = {
        "vectors": vectors, "targets": torch.stack([row["teacher_target"] for row in rows]),
        "groups": [row["id"] for row in rows], "labels": [row["corpus"] for row in rows],
        "active_children": [len(row["lengths"]) for row in rows],
    }
    vector_path = output / "vectors.pt"
    atomic_tensor(vector_path, payload)
    manifest = {
        "study": cfg["study"], "analysis_role": cfg["analysis_role"],
        "config_sha256": cfg["config_sha256"], "source_sha256": source_hashes(),
        "environment": environment(device), "checkpoints": model_provenance,
        "cache_manifest_sha256": sha256(ROOT / cfg["cache_dir"] / "CACHE_MANIFEST.json"),
        "cache_regenerated": True, "original_embedding_bytes_claimed": False,
        "population_order_sha256": stable_hash(payload["groups"]),
        "vectors_sha256": sha256(vector_path), "metadata_sha256": sha256(output / "DOCUMENT_METADATA.json"),
        "timing_protocol": cfg["timing"], "timings": timing,
        "batch_size": cfg["inference"]["batch_size"], "dtype": "float32",
        "documents": len(rows), "merger_updates": 0, "completed_unix": time.time(),
    }
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
    payload["vectors"] = {key: validate_vectors(value.numpy(), key, unit=True)
                          for key, value in payload["vectors"].items()}
    payload["targets"] = validate_vectors(payload["targets"].numpy(), "teacher", unit=True)
    for key in ("groups", "labels", "active_children"):
        payload[key] = np.asarray(payload[key])
    return payload, manifest


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


def evaluate(config_path):
    cfg, official, published = load_protocol(config_path)
    output = ROOT / cfg["output_dir"]
    if (output / "RESULTS.json").exists():
        raise FileExistsError("complete analysis already exists; use report, not another evaluation")
    payload, inference_manifest = load_inference(cfg)
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
        "source_sha256": source_hashes(), "fidelity": fidelity,
        "detectability": aggregate_attacks, "timing": timing,
        "calibration": {"protocol": cfg["calibration"], "permutation_order_sha256": stable_hash(permutations.tolist()), "methods": calibration},
        "published_learned_comparison": comparisons,
        "bootstrap_interpretation": "document-conditional; fixed checkpoints, fitted classifiers, split and selected families; no classifier refit or family reselection in bootstrap",
        "original_results_sha256": require_hash(ROOT / cfg["published_results"], PUBLISHED_HASH),
        "attack_result_sha256": {str(path.relative_to(output)): sha256(path) for path in sorted((output / "attacks").glob("*.json"))},
        "completed_unix": time.time(), "merger_updates": 0,
    }
    atomic_json(output / "RESULTS.json", result)
    render(config_path)


def render(config_path):
    cfg, _, _ = load_protocol(config_path)
    output = ROOT / cfg["output_dir"]
    result = json.loads((output / "RESULTS.json").read_text())
    inference = json.loads((output / "INFERENCE_MANIFEST.json").read_text())
    means = result["fidelity"]["three_seed_mean"]["pooled"]["methods"]
    lines = ["# Matched safe-arm baselines", "", "Additive post-review analysis of the previously evaluated 4,000 documents. All methods use the same regenerated FP32 leaves and direct teacher targets; no merger retraining.", "",
             "| Representation | Teacher cosine | Delta vs weighted | Delta vs unweighted | Source AUC [95% CI] | ms/document |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    tex = [r"% Generated by scripts/evaluate_safe_baselines.py report; do not edit numbers.",
           r"\begin{table}[H]", r"\caption{Matched teacher-comparable prefix comparison on regenerated inputs for the same 4,000 documents. Learned rows average the three final checkpoints; deterministic baselines are evaluated once. $\Delta_w$ and $\Delta_u$ are paired mean teacher-cosine differences against weighted and unweighted pooling. Timings measure device-resident composition only; reference extraction uses the complete forward pass, not an optimized reference-only implementation.}", r"\label{tab:safe-baselines}", r"\centering\small", r"\begin{tabularx}{\linewidth}{Yrrrrr}", r"\toprule",
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
                  f"- GPU: {inference['environment']['gpu']}",
                  f"- Timing: {cfg['timing']['boundary']}; float32; batch size {cfg['inference']['batch_size']}; {cfg['timing']['warmup_passes']} warm-up and {cfg['timing']['measured_passes']} measured passes; TF32 disabled.",
                  "- Reference and parent timing use the same measured full-forward call; reference-only speed is not claimed.",
                  "- No teacher encoding, cache loading, transfers, decoder evaluation, or classifier fitting is included in these inference timings.",
                  "- Original official results remain unchanged. Regenerated tensor byte identity is not claimed; preparation uses safe-only rather than full-document leaf batches.",
                  "- Confidence intervals condition on fixed checkpoints, fitted classifiers, split, and selected classifier families. They do not establish downstream utility or uncertainty over arbitrary training seeds.",
                  "", "## Comparison with published learned results", "",
                  "Tolerance checks diagnose numerical reproducibility, not scientific acceptance; all differences are retained.", "",
                  "| Seed | Scope | Cosine difference | AUC difference | Within tolerances |", "| --- | --- | ---: | ---: | --- |"])
    for seed, scopes in result["published_learned_comparison"].items():
        for scope, entry in scopes.items():
            passed = entry["cosine_within_declared_tolerance"] and entry["auc_within_declared_tolerance"]
            lines.append(f"| {seed} | {scope} | {entry['teacher_cosine_delta']:+.9f} | {entry['auc_delta']:+.9f} | {passed} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "safe_baselines_table.tex").write_text("\n".join(tex) + "\n")
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "ARTIFACT_HASHES.json"]
    atomic_json(output / "ARTIFACT_HASHES.json", {str(path.relative_to(output)): sha256(path) for path in sorted(files)})
    print(f"Report written: {output / 'REPORT.md'}", flush=True)


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
