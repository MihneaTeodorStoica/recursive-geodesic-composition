"""Validation-only comparison of arbitrary-arity geodesic encoders.

Every arm is evaluated on the candidate config's exact ``validation.pt`` rows.
The legacy binary-vs-K4 command-line defaults and JSON labels remain supported,
but explicit reference/candidate checkpoints are labeled by their actual arity.
There is deliberately no fallback to any other data role.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .architecture import GeodesicMerge, contiguous_reduction_groups
from .evaluate import evaluate_attacks
from .train import _merge_level, _root_target, _validate_cache_arity

LEGACY_CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")
FORBIDDEN = ("test", "official")


def reject_forbidden_path(path):
    if any(word in str(path).lower() for word in FORBIDDEN):
        raise ValueError(f"refusing forbidden evaluation artifact path: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(checkpoint_path, device):
    reject_forbidden_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    state = checkpoint["state_dict"]
    dimension = state["input.0.weight"].shape[1] - 2
    model = GeodesicMerge(
        d=dimension,
        hidden=int(cfg["encoder_hidden"]),
        layers=int(cfg["encoder_layers"]),
        heads=int(cfg["heads"]),
        max_angle=float(cfg.get("max_geodesic_angle", cfg.get("max_angle", 0.8))),
        decoder=checkpoint.get("variant") == "autoencoder",
        decoder_layers=int(cfg.get("decoder_layers", 1)),
        decoder_length_mode=str(cfg.get("decoder_length_mode", "oracle")),
        importance_mode=str(cfg.get("importance_mode", "token")),
        importance_beta=float(cfg.get("importance_beta", 1.0)),
        decoder_mode=str(cfg.get("decoder_mode", "legacy")),
        geodesic_parameterization=str(cfg.get("geodesic_parameterization", "legacy")),
        arity=int(cfg.get("arity", 2)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def reduction_depth(leaves, arity):
    depth = 0
    while leaves > 1:
        leaves = len(contiguous_reduction_groups(leaves, arity))
        depth += 1
    return depth


@torch.inference_mode()
def roots_for_rows(model, rows, device, batch_size, collect_angles=True):
    """Batched roots in input order with all-level saturation telemetry."""
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["leaves"].shape[0])].append((index, row))
    output = [None] * len(rows)
    saturated = total_angles = 0
    model_dtype = next(model.parameters()).dtype
    for group in grouped.values():
        for start in range(0, len(group), batch_size):
            part = group[start : start + batch_size]
            # K-ary caches may intentionally store leaves in fp16. Cast only at
            # the model boundary so fp16 storage remains evaluation-compatible.
            nodes = torch.stack([row["leaves"] for _, row in part]).to(
                device=device, dtype=model_dtype)
            sizes = torch.stack([row["lengths"] for _, row in part]).to(
                device=device, dtype=model_dtype)
            while nodes.shape[1] > 1:
                nodes, sizes, _, _, _, _ = _merge_level(model, nodes, sizes)
                if collect_angles:
                    aux = model._last_merge_aux
                    saturated += int(aux["geodesic_saturated"].sum().item())
                    total_angles += int(aux["geodesic_saturated"].numel())
            for (index, _), root in zip(part, nodes[:, 0].float().cpu()):
                output[index] = root
    return torch.stack(output), saturated / max(total_angles, 1)


def corpus_ids(cfg):
    return tuple(str(item["id"]) for item in cfg.get("corpora", ())) or LEGACY_CORPORA


def load_role(cfg, role):
    if role not in ("train", "validation"):
        raise ValueError("K-ary evaluator accepts only train/validation roles")
    cache = Path(cfg["cache_dir"])
    reject_forbidden_path(cache)
    rows_by = {}
    for corpus in corpus_ids(cfg):
        path = cache / corpus / f"{role}.pt"
        sidecar_path = Path(str(path) + ".json")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("split") != role or sidecar.get("official_test_evaluated") is not False:
            raise ValueError(f"cache is not explicit validation-only K-ary data: {path}")
        rows = torch.load(path, map_location="cpu", weights_only=False)
        _validate_cache_arity(rows, int(cfg["arity"]), f"{role}/{corpus}")
        rows_by[corpus] = rows
    return rows_by


def load_validation(cfg):
    """Backward-compatible public helper."""
    return load_role(cfg, "validation")


def segmentation_consistency(rows, roots):
    grouped = defaultdict(list)
    for row, root in zip(rows, roots):
        grouped[str(row["id"])].append(root)
    values = []
    for variants in grouped.values():
        if len(variants) >= 2:
            values.append(float(F.cosine_similarity(variants[0][None], variants[1][None]).item()))
    return {"documents": len(values), "mean_cosine": float(np.mean(values)) if values else None,
            "minimum_cosine": float(np.min(values)) if values else None}


def _orientation_safe_attacks(real, fake, ids):
    attacks = evaluate_attacks(real, fake, np.asarray(ids), bootstrap=0)
    for metrics in attacks["attacks"].values():
        metrics["orientation_safe_auc"] = max(float(metrics["auc"]), 1.0 - float(metrics["auc"]))
    strongest = max(attacks["attacks"],
                    key=lambda name: attacks["attacks"][name]["orientation_safe_auc"])
    attacks["orientation_safe_strongest_attack"] = strongest
    attacks["orientation_safe_max_attack_auc"] = attacks["attacks"][strongest]["orientation_safe_auc"]
    return attacks


def quality_metrics(model, rows_by, device, cfg):
    all_fake, all_real, all_ids, all_rows = [], [], [], []
    per_corpus = {}
    saturated_count = angle_count = 0
    batch_size = int(cfg.get("validation_root_batch_size", 64))
    for corpus, rows in rows_by.items():
        roots, saturation = roots_for_rows(model, rows, device, batch_size)
        targets = torch.stack([_root_target(row) for row in rows]).float()
        cosines = F.cosine_similarity(roots, targets, dim=-1)
        corpus_attacks = _orientation_safe_attacks(targets.numpy(), roots.numpy(),
                                                   [row["id"] for row in rows])
        per_corpus[corpus] = {
            "documents": len(rows),
            "teacher_cosine": float(cosines.mean()),
            "attacks": corpus_attacks,
            "logistic_auc": corpus_attacks["attacks"]["standard_scaled_logistic"]["auc"],
            "rbf_auc": corpus_attacks["attacks"]["standard_scaled_rbf_svm"]["auc"],
            "orientation_safe_strongest_auc": corpus_attacks["orientation_safe_max_attack_auc"],
            "segmentation_consistency": segmentation_consistency(rows, roots),
        }
        all_fake.append(roots.numpy()); all_real.append(targets.numpy())
        all_ids.extend(row["id"] for row in rows); all_rows.extend(rows)
        # roots_for_rows exposes a fraction; retain exact weighting across corpora
        # by using the number of reduction calls as a stable denominator proxy.
        corpus_angles = sum(max(1, reduction_depth(int(row["leaves"].shape[0]), model.arity))
                            for row in rows)
        saturated_count += saturation * corpus_angles
        angle_count += corpus_angles
    fake, real = np.concatenate(all_fake), np.concatenate(all_real)
    attacks = _orientation_safe_attacks(real, fake, all_ids)
    depths = [reduction_depth(int(row["leaves"].shape[0]), model.arity) for row in all_rows]
    return {
        "per_corpus": per_corpus,
        "macro_teacher_cosine": float(np.mean([v["teacher_cosine"] for v in per_corpus.values()])),
        "attacks": attacks,
        "strongest_attack_auc": attacks["max_attack_auc"],  # compatibility
        "orientation_safe_strongest_attack_auc": attacks["orientation_safe_max_attack_auc"],
        "geodesic_saturation_fraction": float(saturated_count / max(angle_count, 1)),
        "tree_depth_mean": float(np.mean(depths)),
        "tree_depth_max": int(max(depths)),
    }


def load_long_validation(cache_dir, corpora=LEGACY_CORPORA):
    root = Path(cache_dir)
    reject_forbidden_path(root)
    rows = []
    for corpus in corpora:
        path = root / f"{corpus}.pt"
        corpus_rows = torch.load(path, map_location="cpu", weights_only=False)
        for row in corpus_rows:
            if "targets" in row or int(row.get("token_count", 0)) <= 6144:
                raise ValueError(f"invalid long validation row: {path}")
        rows.extend(corpus_rows)
    return rows


def benchmark_roots(model, rows, device, batch_size, warmup=2, repeats=5):
    for _ in range(warmup):
        roots_for_rows(model, rows, device, batch_size, collect_angles=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        roots_for_rows(model, rows, device, batch_size, collect_angles=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    seconds = float(np.median(elapsed))
    leaves = sum(int(row["leaves"].shape[0]) for row in rows)
    return {"documents": len(rows), "leaves": leaves, "median_seconds": seconds,
            "documents_per_second": len(rows) / seconds,
            "leaves_per_second": leaves / seconds,
            "batch_size": batch_size, "device": str(device), "warmup": warmup,
            "repeats": repeats}


class ParentOnlyRetentionProbe(torch.nn.Module):
    """Matched ordered K-child probe whose only input is the frozen parent."""
    def __init__(self, dimension, arity, hidden):
        super().__init__()
        self.dimension, self.arity = int(dimension), int(arity)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(dimension), torch.nn.Linear(dimension, hidden),
            torch.nn.GELU(), torch.nn.Linear(hidden, arity * dimension))
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, parent):
        parent = F.normalize(parent.float(), dim=-1)
        delta = self.network(parent).reshape(-1, self.arity, self.dimension)
        return F.normalize(parent[:, None, :] + .25 * torch.tanh(delta) /
                           np.sqrt(self.dimension), dim=-1)


def _exact_arity_rows(rows_by, arity, per_corpus):
    selected, labels = [], []
    for corpus, rows in rows_by.items():
        matches = [row for row in rows if int(row["leaves"].shape[0]) == int(arity)]
        selected.extend(matches[:int(per_corpus)])
        labels.extend([corpus] * min(len(matches), int(per_corpus)))
    return selected, labels


@torch.inference_mode()
def _probe_features(model, rows, device, batch_size):
    roots, _ = roots_for_rows(model, rows, device, batch_size, collect_angles=False)
    children = torch.stack([row["leaves"].float() for row in rows])
    return roots, children


def _fit_retention_probe(model, train_rows, validation_rows, validation_labels,
                         device, cfg, destination):
    arity = int(cfg["retention_probe_arity"])
    batch_size = int(cfg.get("retention_probe_batch_size", 128))
    roots_train, children_train = _probe_features(
        model, train_rows, device, int(cfg.get("validation_root_batch_size", 32)))
    roots_validation, children_validation = _probe_features(
        model, validation_rows, device, int(cfg.get("validation_root_batch_size", 32)))
    seed = int(cfg.get("retention_probe_seed", 730241))
    torch.manual_seed(seed)
    probe = ParentOnlyRetentionProbe(model.d, arity,
                                     int(cfg.get("retention_probe_hidden", 512))).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(),
                                  lr=float(cfg.get("retention_probe_learning_rate", 3e-4)),
                                  weight_decay=float(cfg.get("retention_probe_weight_decay", 1e-3)))
    roots_train, children_train = roots_train.to(device), children_train.to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    updates = int(cfg.get("retention_probe_updates", 400))
    probe.train()
    final_loss = None
    for _ in range(updates):
        indexes = torch.randint(len(train_rows), (min(batch_size, len(train_rows)),),
                                generator=generator)
        indexes_device = indexes.to(device)
        prediction = probe(roots_train[indexes_device])
        target = children_train[indexes_device]
        cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
        loss = cosine + float(cfg.get("retention_probe_mse_weight", .25)) * F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final_loss = float(loss.detach())
    probe.eval()
    with torch.inference_mode():
        prediction = probe(roots_validation.to(device)).cpu()
    cosine = F.cosine_similarity(prediction, children_validation, dim=-1)
    per_corpus = {}
    labels = np.asarray(validation_labels)
    for corpus in sorted(set(validation_labels)):
        mask = torch.from_numpy(labels == corpus)
        per_corpus[corpus] = {"rows": int(mask.sum()), "child_cosine": float(cosine[mask].mean())}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.cpu().state_dict(), "arity": arity,
                "dimension": model.d, "seed": seed, "updates": updates,
                "validation_only": True, "official_test_evaluated": False}, destination)
    return {"available": True, "arity": arity, "input": "parent_only",
            "ordered_children": arity, "train_rows": len(train_rows),
            "validation_rows": len(validation_rows), "updates": updates,
            "batch_size": batch_size, "final_training_loss": final_loss,
            "child_cosine": float(cosine.mean()), "per_corpus": per_corpus,
            "checkpoint": str(destination), "checkpoint_sha256": sha256_file(destination)}


def matched_retention_probes(models, train_by, validation_by, device, cfg, output_dir):
    arity = int(cfg.get("retention_probe_arity", cfg["arity"]))
    train_rows, _ = _exact_arity_rows(train_by, arity,
                                      int(cfg.get("retention_probe_train_rows_per_corpus", 512)))
    validation_rows, validation_labels = _exact_arity_rows(
        validation_by, arity, int(cfg.get("retention_probe_validation_rows_per_corpus", 256)))
    minimum = int(cfg.get("retention_probe_minimum_rows", 32))
    if len(train_rows) < minimum or len(validation_rows) < minimum:
        return {"available": False,
                "reason": f"insufficient exact-{arity}-child rows: train={len(train_rows)}, validation={len(validation_rows)}",
                "arms": {}}
    arms = {}
    for label, model in models.items():
        arms[label] = _fit_retention_probe(
            model, train_rows, validation_rows, validation_labels, device, cfg,
            Path(output_dir) / "retention_probes" / f"parent_only_{label}.pt")
    return {"available": True, "protocol": "matched_parent_only_ordered_children_v1",
            "arity": arity, "same_rows_all_arms": True, "arms": arms}


def _mean_segmentation(metrics):
    values = [entry["segmentation_consistency"]["mean_cosine"]
              for entry in metrics["per_corpus"].values()
              if entry["segmentation_consistency"]["mean_cosine"] is not None]
    return float(np.mean(values)) if values else None


def escalation_gates(reference, candidate, retention=None, reference_label=None, candidate_label=None):
    """K-independent comparison; legacy callers may omit retention/labels."""
    auc_key = "orientation_safe_strongest_attack_auc"
    reference_auc = reference.get(auc_key, reference["strongest_attack_auc"])
    candidate_auc = candidate.get(auc_key, candidate["strongest_attack_auc"])
    auc_ok = candidate_auc <= reference_auc + 0.01
    cosine_ok = (candidate["macro_teacher_cosine"] >= 0.97 and
                 candidate["macro_teacher_cosine"] >= reference["macro_teacher_cosine"] - 0.002 and
                 min(x["teacher_cosine"] for x in candidate["per_corpus"].values()) >= 0.97)
    saturation_ok = (candidate["geodesic_saturation_fraction"] == 0.0 and
                     candidate.get("training_saturation_guard_passed", True))
    depth_reduction = 1.0 - candidate["tree_depth_mean"] / reference["tree_depth_mean"]
    throughput_key = ("validation_root_throughput"
                      if "validation_root_throughput" in candidate else "long_root_throughput")
    speedup = (candidate[throughput_key]["documents_per_second"] /
               reference[throughput_key]["documents_per_second"])
    segmentation_reference = _mean_segmentation(reference)
    segmentation_candidate = _mean_segmentation(candidate)
    segmentation_ok = (segmentation_reference is not None and segmentation_candidate is not None and
                       segmentation_candidate >= segmentation_reference - .002)
    retention = retention or {"available": False, "arms": {},
                              "reason": "matched parent-only retention probe unavailable"}
    retention_ok = False
    if retention.get("available") and reference_label in retention.get("arms", {}) and candidate_label in retention.get("arms", {}):
        retention_reference = retention["arms"][reference_label]["child_cosine"]
        retention_candidate = retention["arms"][candidate_label]["child_cosine"]
        retention_ok = retention_candidate >= retention_reference - .01
    legacy_k4 = reference.get("arity") == 2 and candidate.get("arity") == 4
    if legacy_k4:
        # Exact historical K2->K4 names/thresholds for existing report readers.
        gates = {
            "strongest_auc_noninferior": auc_ok,
            "cosine_noninferior_and_floor": cosine_ok,
            "no_angle_saturation": saturation_ok,
            "depth_reduction_at_least_25pct": depth_reduction >= 0.25,
            "long_root_speedup_at_least_15pct": speedup >= 1.15,
            "retention_noninferior": retention_ok,
        }
        return {"gates": gates, "depth_reduction_fraction": depth_reduction,
                "long_root_speedup": speedup, "retention": retention,
                "eligible_to_build_k8": all(gates.values()),
                "eligible_to_recommend_next_arity": all(gates.values())}
    # The K8->K16 gate requires another 20% depth reduction and another 15%
    # throughput gain versus K4. Both are reported independently and required.
    gates = {
        "strongest_auc_noninferior": auc_ok,
        "cosine_noninferior_and_floor": cosine_ok,
        "no_angle_saturation": saturation_ok,
        "depth_reduction_at_least_20pct": depth_reduction >= 0.20,
        "validation_root_speedup_at_least_15pct": speedup >= 1.15,
        "segmentation_noninferior": segmentation_ok,
        "retention_noninferior": retention_ok,
    }
    return {"gates": gates, "depth_reduction_fraction": depth_reduction,
            "validation_root_speedup": speedup, "retention": retention,
            "eligible_to_recommend_next_arity": all(gates.values())}


def _arm_label(role, arity, legacy=False):
    if arity == 2:
        return "binary_reference" if role != "candidate" else "binary_candidate"
    if legacy and role == "candidate" and arity == 4:
        return "kary4"
    return f"kary{arity}_{role}" if role == "reference" else f"kary{arity}"


def _artifact_manifest(result, cfg_path, output_path, checkpoints):
    artifacts = {"config": {"path": str(cfg_path), "sha256": sha256_file(cfg_path)},
                 "evaluation": {"path": str(output_path), "sha256": sha256_file(output_path)}}
    for label, path in checkpoints.items():
        artifacts[f"checkpoint/{label}"] = {"path": str(path), "sha256": sha256_file(path)}
    for label, metrics in result["models"].items():
        probe = result.get("retention", {}).get("arms", {}).get(label)
        if probe and probe.get("checkpoint"):
            artifacts[f"retention_probe/{label}"] = {
                "path": probe["checkpoint"], "sha256": probe["checkpoint_sha256"]}
    return {"validation_only": True, "official_test_evaluated": False,
            "cache_manifest": str(Path(result["cache_dir"]) / "CACHE_MANIFEST.json"),
            "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description="Validation-only arbitrary K-ary screen")
    parser.add_argument("--config", default="configs/kary4_geodesic_encoder.yaml")
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--candidate-checkpoint", default=None)
    # Legacy aliases preserve old invocations/results labeling.
    parser.add_argument("--binary-checkpoint", default=None)
    parser.add_argument("--kary-checkpoint", default=None)
    parser.add_argument("--optional-binary-checkpoint", default=None)
    parser.add_argument("--long-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg.get("validation_only") is not True or cfg.get("official_test_evaluated") is not False:
        raise ValueError("config must be explicit validation-only/unofficial")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows_by = load_validation(cfg)
    train_by = load_role(cfg, "train") if bool(cfg.get("retention_probe_enabled", False)) else None
    explicit = args.reference_checkpoint is not None or args.candidate_checkpoint is not None
    reference_path = (args.reference_checkpoint or args.binary_checkpoint or
                      "results_geodesic_tangent_angle05_equal_encoder_1500/encoder_seed424242.pt")
    candidate_path = (args.candidate_checkpoint or args.kary_checkpoint or
                      "results_kary4_geodesic_encoder/encoder_seed424242.pt")
    loaded = []
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        model, checkpoint = build_model(path, device)
        label = _arm_label(role, model.arity, legacy=not explicit)
        loaded.append((label, role, Path(path), model, checkpoint))
    if args.optional_binary_checkpoint:
        model, checkpoint = build_model(args.optional_binary_checkpoint, device)
        loaded.append((_arm_label("reference", model.arity), "optional_reference",
                       Path(args.optional_binary_checkpoint), model, checkpoint))
    labels = [item[0] for item in loaded]
    if len(labels) != len(set(labels)):
        raise ValueError(f"ambiguous arm labels after reading actual checkpoint arities: {labels}")

    output = Path(args.output or (Path(cfg["output_dir"]) / "KARY_VALIDATION_REPORT.json"))
    reject_forbidden_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_rows = [row for rows in rows_by.values() for row in rows]
    long_cache = args.long_cache
    if not explicit and long_cache is None:
        long_cache = "cache_information_retention_long_validation"
    if long_cache:
        benchmark_rows = load_long_validation(long_cache, corpus_ids(cfg))
    result = {"validation_only": True, "official_test_evaluated": False,
              "cache_dir": cfg["cache_dir"], "same_validation_rows_all_arms": True,
              "throughput_rows": "long_validation" if long_cache else "exact_candidate_validation_cache",
              "models": {}, "model_roles": {}}
    model_map = {}
    checkpoints = {}
    for label, role, path, model, checkpoint in loaded:
        metrics = quality_metrics(model, rows_by, device, cfg)
        metrics["arity"] = model.arity
        metrics["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        metrics["selected_step"] = checkpoint.get("selected_step")
        status_path = path.with_name(path.stem + "_run_status.json")
        run_status = json.loads(status_path.read_text()) if status_path.exists() else None
        metrics["training_run_status"] = run_status
        rejection = (run_status or {}).get("rejection_reason") or ""
        metrics["training_saturation_guard_passed"] = not rejection.startswith(
            "geodesic_saturation_nonzero")
        metrics["training_selection"] = {
            "metric": checkpoint.get("checkpoint_selection_metric"),
            "raw": checkpoint.get("checkpoint_selection_raw_value"),
            "moving_average": checkpoint.get("checkpoint_selection_smoothed_value"),
            "window": checkpoint.get("checkpoint_moving_average_window"),
        }
        metrics["validation_root_throughput"] = benchmark_roots(
            model, benchmark_rows, device, int(cfg.get("validation_root_batch_size", 64)),
            warmup=int(cfg.get("throughput_warmup", 2)),
            repeats=int(cfg.get("throughput_repeats", 5)))
        # Keep the established field available to report consumers.
        metrics["long_root_throughput"] = metrics["validation_root_throughput"]
        result["models"][label] = metrics
        result["model_roles"][role] = label
        model_map[label] = model
        checkpoints[label] = path

    if train_by is not None:
        result["retention"] = matched_retention_probes(
            model_map, train_by, rows_by, device, cfg, output.parent)
    else:
        result["retention"] = {"available": False, "reason": "disabled by config", "arms": {}}
    reference_label = result["model_roles"]["reference"]
    candidate_label = result["model_roles"]["candidate"]
    result["comparison"] = escalation_gates(
        result["models"][reference_label], result["models"][candidate_label],
        result["retention"], reference_label, candidate_label)
    # Existing consumers expect ``escalation``. Keep it as an exact alias.
    result["escalation"] = result["comparison"]
    output.write_text(json.dumps(result, indent=2) + "\n")
    manifest_path = output.parent / "ARTIFACT_MANIFEST.json"
    manifest = _artifact_manifest(result, cfg_path, output, checkpoints)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
