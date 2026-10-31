"""Focused K4 TRAIN/validation deployment-regime study.

The loader is fail-closed: it accepts only the configured TRAIN and validation
cache files, rejects forbidden path labels, and enforces source ranges below the
configured exclusive ceiling before loading tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate_kary_validation import ParentOnlyRetentionProbe, build_model
from .train import _root_target, _validate_cache_arity
from .validation_output_calibration import AffineCalibrator, cosine_mean, diagnostics, sha256_file

CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")
LEAF_COUNTS = (2, 3, 4)
ROLES = ("train", "validation")
FORBIDDEN_PATH_TERMS = ("official", "test")


def reject_path(path: str | Path) -> None:
    text = str(path).lower()
    if any(term in text for term in FORBIDDEN_PATH_TERMS):
        raise ValueError(f"refusing forbidden artifact path: {path}")


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(cfg: dict) -> None:
    if cfg.get("validation_only") is not True or tuple(cfg.get("roles", ())) != ROLES:
        raise ValueError("study must declare exactly TRAIN and validation roles")
    if tuple(cfg.get("corpora", ())) != CORPORA or tuple(cfg.get("leaf_counts", ())) != LEAF_COUNTS:
        raise ValueError("corpus or child-count declaration changed")
    if int(cfg["maximum_source_index_exclusive"]) != 20000:
        raise ValueError("source ceiling must remain 20000 exclusive")
    for role in ROLES:
        source = cfg["source_ranges"][role]
        start, count = int(source["start"]), int(source["count"])
        if start < 0 or count < 1 or start + count > int(cfg["maximum_source_index_exclusive"]):
            raise ValueError(f"forbidden source range for {role}: [{start},{start + count})")
    ft = cfg["fine_tuning"]
    if int(ft["steps"]) > 500 or int(ft["steps"]) < 1:
        raise ValueError("focused recipe requires 1..500 steps")
    if not 0.0 < float(ft["learning_rate"]) <= 2e-5:
        raise ValueError("focused learning rate exceeds 2e-5")
    if int(ft["validation_every"]) != 100 or int(ft["moving_average_window"]) != 3:
        raise ValueError("validation/MA3 declaration changed")
    if int(cfg["fit_groups_per_corpus_arity"]) * len(CORPORA) * len(LEAF_COUNTS) <= 768:
        raise ValueError("full-dimensional Procrustes needs more fit rows than dimensions")
    for key in ("cache_dir", "output_dir", "attack_config"):
        reject_path(cfg[key])
    seen = []
    for entry in cfg["seeds"]:
        seen.append(int(entry["seed"]))
        reject_path(entry["checkpoint"])
        reject_path(entry["frozen_probe"])
    if seen != [424242, 271828, 314159]:
        raise ValueError("seed screen order changed")


def verify_file(root: Path, relative: str, expected_hash: str) -> Path:
    reject_path(relative)
    path = root / relative
    reject_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {relative}: {actual}")
    return path


def direct_target(row: dict) -> torch.Tensor:
    target = _root_target(row)
    if target is None or not torch.isfinite(target).all():
        raise ValueError(f"missing/non-finite direct root target for {row.get('id')}")
    return target.float()


def row_is_eligible(row: dict, cfg: dict, role: str) -> bool:
    leaves = int(row["leaves"].shape[0])
    tokens = int(row["lengths"].sum().item())
    if leaves not in LEAF_COUNTS or not int(cfg["minimum_prefix_tokens"]) <= tokens <= int(cfg["maximum_prefix_tokens"]):
        return False
    if role == "validation" and row.get("segmentation") != cfg["validation_segmentation"]:
        return False
    if role == "train" and row.get("segmentation") != "train":
        return False
    try:
        direct_target(row)
    except ValueError:
        return False
    return True


def load_caches(root: Path, cfg: dict) -> tuple[dict, dict]:
    cache_root = root / cfg["cache_dir"]
    reject_path(cache_root)
    loaded: dict[str, dict[str, list]] = {role: {} for role in ROLES}
    provenance = {}
    ceiling = int(cfg["maximum_source_index_exclusive"])
    for corpus in CORPORA:
        for role in ROLES:
            path = cache_root / corpus / f"{role}.pt"
            sidecar_path = Path(str(path) + ".json")
            reject_path(path)
            sidecar = json.loads(sidecar_path.read_text())
            expected_range = cfg["source_ranges"][role]
            expected = {
                "cache_namespace": "kary_geodesic_v1",
                "split": role,
                "arity": int(cfg["arity"]),
                "source_start": int(expected_range["start"]),
                "requested": int(expected_range["count"]),
                "corpus_id": corpus,
            }
            for key, value in expected.items():
                if sidecar.get(key) != value:
                    raise ValueError(f"cache provenance mismatch {corpus}/{role}/{key}")
            source_end = int(sidecar["source_start"]) + int(sidecar["requested"])
            if int(sidecar["source_start"]) < 0 or source_end > ceiling:
                raise ValueError(f"source range reaches forbidden index for {corpus}/{role}")
            rows = torch.load(path, map_location="cpu", weights_only=False)
            _validate_cache_arity(rows, int(cfg["arity"]), f"{role}/{corpus}")
            if int(sidecar.get("usable", -1)) != len(rows):
                raise ValueError(f"row-count mismatch for {corpus}/{role}")
            for row in rows:
                if not torch.isfinite(row["leaves"]).all() or not torch.isfinite(row["lengths"]).all():
                    raise FloatingPointError(f"non-finite cache row in {corpus}/{role}")
            loaded[role][corpus] = rows
            provenance[f"{corpus}/{role}"] = {
                **expected,
                "source_end_exclusive": source_end,
                "usable": len(rows),
                "tensor_path": str(path.relative_to(root)),
                "tensor_sha256": sha256_file(path),
                "sidecar_path": str(sidecar_path.relative_to(root)),
                "sidecar_sha256": sha256_file(sidecar_path),
                "range_guard_passed": True,
            }
        train_ids = {str(row["id"]) for row in loaded["train"][corpus]}
        validation_ids = {str(row["id"]) for row in loaded["validation"][corpus]}
        if train_ids & validation_ids:
            raise AssertionError(f"TRAIN/validation source-document overlap in {corpus}")
    return loaded, provenance


def eligible_records(rows_by: dict, cfg: dict, role: str) -> list[dict]:
    records = []
    for corpus in CORPORA:
        for index, row in enumerate(rows_by[corpus]):
            if row_is_eligible(row, cfg, role):
                records.append({
                    "corpus": corpus,
                    "role": role,
                    "row_index": index,
                    "group": str(row["id"]),
                    "leaf_count": int(row["leaves"].shape[0]),
                    "tokens": int(row["lengths"].sum().item()),
                    "row": row,
                })
    return records


def characterize_records(records: list[dict]) -> dict:
    result = {"rows": len(records), "source_documents": len({(r['corpus'], r['group']) for r in records}), "by_corpus_arity": {}}
    for corpus in CORPORA:
        result["by_corpus_arity"][corpus] = {}
        for leaves in LEAF_COUNTS:
            part = [r for r in records if r["corpus"] == corpus and r["leaf_count"] == leaves]
            result["by_corpus_arity"][corpus][str(leaves)] = {
                "rows": len(part),
                "source_documents": len({r["group"] for r in part}),
                "minimum_tokens": min((r["tokens"] for r in part), default=None),
                "maximum_tokens": max((r["tokens"] for r in part), default=None),
            }
    return result


def select_balanced_fit(records: list[dict], count: int, fit_seed: int) -> list[dict]:
    selected = []
    for corpus in CORPORA:
        used_groups: set[str] = set()
        # Scarce four-child documents are assigned first; all cells remain disjoint.
        for leaves in reversed(LEAF_COUNTS):
            by_group = {}
            for record in records:
                if record["corpus"] == corpus and record["leaf_count"] == leaves and record["group"] not in used_groups:
                    by_group.setdefault(record["group"], record)
            ranked = sorted(by_group, key=lambda group: hashlib.sha256(
                f"{fit_seed}|{corpus}|{leaves}|{group}".encode()).digest())
            if len(ranked) < count:
                raise ValueError(f"insufficient disjoint TRAIN groups for {corpus}/{leaves}: {len(ranked)}")
            chosen = ranked[:count]
            selected.extend(by_group[group] for group in chosen)
            used_groups.update(chosen)
    selected.sort(key=lambda r: (CORPORA.index(r["corpus"]), r["leaf_count"], r["group"]))
    if len({(r["corpus"], r["group"]) for r in selected}) != len(selected):
        raise AssertionError("balanced fit contains repeated source documents")
    return selected


def _merge_records(model, records: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    outputs, targets, weights = [], [], []
    angle_values, importance_saturated, importance_total = [], 0, 0
    model_dtype = next(model.parameters()).dtype
    for leaves in LEAF_COUNTS:
        part = [r for r in records if r["leaf_count"] == leaves]
        if not part:
            continue
        children = torch.stack([r["row"]["leaves"].float() for r in part]).to(device=device, dtype=model_dtype)
        masses = torch.stack([r["row"]["lengths"].float() for r in part]).to(device=device, dtype=model_dtype)
        if leaves < int(model.arity):
            children = F.pad(children, (0, 0, 0, int(model.arity) - leaves))
            masses = F.pad(masses, (0, int(model.arity) - leaves))
        parent = model.merge(children, masses)
        aux = model._last_merge_aux
        outputs.append(parent.float())
        targets.append(torch.stack([direct_target(r["row"]) for r in part]).to(device))
        weights.append(aux["effective_weights"].float())
        angle_values.append(aux["angle"].float().reshape(-1))
        active = aux["active_mask"]
        importance_saturated += int(((aux["corrections"].abs() >= .95 * float(model.importance_beta)) & active).sum().item())
        importance_total += int(active.sum().item())
    return (
        torch.cat(outputs), torch.cat(targets), torch.cat(weights),
        {
            "geodesic_saturation_count": int(sum(int((x >= .95 * float(model.max_angle)).sum().item()) for x in angle_values)),
            "geodesic_angle_max": float(torch.cat(angle_values).max().detach()),
            "importance_saturation_count": importance_saturated,
            "importance_active_count": importance_total,
        },
    )


@torch.inference_mode()
def extract_direct(model, records: list[dict], device: torch.device, batch_size: int) -> dict:
    model.eval()
    root_by, target_by, weight_by = {}, {}, {}
    telemetry = {"geodesic_saturation_count": 0, "importance_saturation_count": 0, "importance_active_count": 0, "geodesic_angle_max": 0.0}
    for leaves in LEAF_COUNTS:
        indexes = [i for i, record in enumerate(records) if record["leaf_count"] == leaves]
        for start in range(0, len(indexes), batch_size):
            part_indexes = indexes[start:start + batch_size]
            part = [records[i] for i in part_indexes]
            roots, targets, weights, health = _merge_records(model, part, device)
            for local, index in enumerate(part_indexes):
                root_by[index] = roots[local].cpu().numpy().astype(np.float32)
                target_by[index] = targets[local].cpu().numpy().astype(np.float32)
                weight_by[index] = weights[local].cpu().numpy().astype(np.float32)
            for key in ("geodesic_saturation_count", "importance_saturation_count", "importance_active_count"):
                telemetry[key] += int(health[key])
            telemetry["geodesic_angle_max"] = max(telemetry["geodesic_angle_max"], float(health["geodesic_angle_max"]))
    return {
        "roots": np.stack([root_by[i] for i in range(len(records))]),
        "targets": np.stack([target_by[i] for i in range(len(records))]),
        "weights": np.stack([weight_by[i] for i in range(len(records))]),
        "groups": np.asarray([r["group"] for r in records]),
        "corpora": np.asarray([r["corpus"] for r in records]),
        "leaf_counts": np.asarray([r["leaf_count"] for r in records]),
        "telemetry": telemetry,
    }


def fit_procrustes(predicted: np.ndarray, teacher: np.ndarray, fit_groups: np.ndarray, validation_groups: np.ndarray) -> AffineCalibrator:
    x, y = np.asarray(predicted, np.float64), np.asarray(teacher, np.float64)
    if x.shape != y.shape or x.ndim != 2 or len(x) <= x.shape[1]:
        raise ValueError("invalid full-dimensional Procrustes arrays")
    overlap = set(map(str, fit_groups)) & set(map(str, validation_groups))
    if overlap:
        raise AssertionError("TRAIN/validation fit-group overlap")
    mx, my = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - mx).T @ (y - my), full_matrices=False)
    matrix = u @ vt
    bias = my - mx @ matrix
    metadata = {
        "recipe": "global_full_orthogonal_procrustes_with_train_mean_alignment",
        "fit_role": "train",
        "fit_pairs": len(x),
        "fit_groups": len(set(map(str, fit_groups))),
        "evaluation_role": "validation",
        "evaluation_groups": len(set(map(str, validation_groups))),
        "group_overlap": 0,
        "fit_group_sha256": stable_hash(sorted(map(str, fit_groups))),
        "matrix_orthogonality_error_fro": float(np.linalg.norm(matrix.T @ matrix - np.eye(matrix.shape[0]))),
        "determinant": float(np.linalg.det(matrix)),
        "mean_alignment": True,
        "normalization": "final_l2",
    }
    return AffineCalibrator("deployment_procrustes", matrix, bias, metadata)


def _bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> list[float] | None:
    unique = np.unique(groups)
    if len(unique) < 8 or replicates < 1:
        return None
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([by_group[group] for group in sampled])
        if len(np.unique(y[indexes])) < 2:
            continue
        auc = float(roc_auc_score(y[indexes], score[indexes]))
        values.append(max(auc, 1.0 - auc))
    if len(values) < max(20, replicates // 2):
        return None
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def fixed_attacks(real: np.ndarray, fake: np.ndarray, groups: np.ndarray, attack_cfg: dict, ci_replicates: int, ci_seed: int) -> dict:
    real, fake, groups = np.asarray(real), np.asarray(fake), np.asarray(groups)
    x = np.vstack((real, fake))
    y = np.r_[np.ones(len(real), dtype=int), np.zeros(len(fake), dtype=int)]
    paired_groups = np.r_[groups, groups]
    train, holdout = next(GroupShuffleSplit(
        n_splits=1, test_size=float(attack_cfg["test_size"]), random_state=int(attack_cfg["split_seed"])
    ).split(x, y, paired_groups))
    if set(paired_groups[train]) & set(paired_groups[holdout]):
        raise AssertionError("attack source-document overlap")
    seed = int(attack_cfg["model_seed"])
    models = {
        "standard_scaled_logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=float(attack_cfg["logistic_c"]), max_iter=int(attack_cfg["logistic_max_iter"]), solver="lbfgs", random_state=seed)),
        ]),
        "standard_scaled_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=float(attack_cfg["rbf_c"]), gamma=float(attack_cfg["rbf_gamma"]), kernel="rbf", random_state=seed)),
        ]),
    }
    results = {}
    for offset, (name, model) in enumerate(models.items()):
        model.fit(x[train], y[train])
        score = model.predict_proba(x[holdout])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x[holdout])
        prediction = model.predict(x[holdout])
        raw = float(roc_auc_score(y[holdout], score))
        balanced = float(balanced_accuracy_score(y[holdout], prediction))
        results[name] = {
            "raw_auc": raw,
            "orientation_safe_auc": max(raw, 1.0 - raw),
            "orientation_safe_auc_ci95_group_bootstrap": _bootstrap_ci(
                y[holdout], score, paired_groups[holdout], ci_replicates, ci_seed + offset),
            "pr_auc": float(average_precision_score(y[holdout], score)),
            "balanced_accuracy": balanced,
            "orientation_safe_balanced_accuracy": max(balanced, 1.0 - balanced),
        }
    strongest = max(results, key=lambda name: results[name]["orientation_safe_auc"])
    return {
        "attacks": results,
        "strongest_attack": strongest,
        "strongest_orientation_safe_auc": results[strongest]["orientation_safe_auc"],
        "strongest_orientation_safe_auc_ci95": results[strongest]["orientation_safe_auc_ci95_group_bootstrap"],
        "attack_train_documents": int(len(np.unique(paired_groups[train]))),
        "attack_holdout_documents": int(len(np.unique(paired_groups[holdout]))),
        "group_overlap": 0,
    }


def cosine_ci(fake: np.ndarray, real: np.ndarray, replicates: int, seed: int) -> list[float] | None:
    values = np.sum(F.normalize(torch.from_numpy(fake).float(), dim=1).numpy() * F.normalize(torch.from_numpy(real).float(), dim=1).numpy(), axis=1)
    if len(values) < 8 or replicates < 1:
        return None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(replicates)]
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def evaluate_scopes(fake: np.ndarray, real: np.ndarray, groups: np.ndarray, corpora: np.ndarray, pre_norm: np.ndarray, cfg: dict, attack_cfg: dict) -> dict:
    scopes = {"pooled": np.arange(len(fake))}
    scopes.update({corpus: np.flatnonzero(corpora == corpus) for corpus in CORPORA})
    result = {}
    for offset, (scope, indexes) in enumerate(scopes.items()):
        attack = fixed_attacks(
            real[indexes], fake[indexes], groups[indexes], attack_cfg,
            int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]) + 100 * offset,
        )
        result[scope] = {
            "rows": len(indexes),
            "source_documents": len(set(groups[indexes].tolist())),
            "teacher_cosine": cosine_mean(fake[indexes], real[indexes]),
            "teacher_cosine_ci95_row_bootstrap": cosine_ci(
                fake[indexes], real[indexes], int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]) + 1000 + offset),
            "attacks": attack,
            "diagnostics": diagnostics(fake[indexes], pre_norm[indexes], cfg),
        }
    return result


def save_transform(calibrator: AffineCalibrator, path: Path, root: Path) -> dict:
    info = calibrator.save(path)
    return {
        **info,
        "path": str(path.relative_to(root)),
        "fit_metadata": calibrator.fit_metadata,
        "minimum_singular_value": float(np.linalg.svd(calibrator.matrix, compute_uv=False).min()),
        "maximum_singular_value": float(np.linalg.svd(calibrator.matrix, compute_uv=False).max()),
    }


def load_probe(path: Path, expected_hash: str, dimension: int, device: torch.device) -> tuple[ParentOnlyRetentionProbe, dict]:
    if sha256_file(path) != expected_hash:
        raise ValueError("frozen probe hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint["arity"]) != 4 or int(checkpoint["dimension"]) != dimension or checkpoint.get("fit_role") != "train":
        raise ValueError("frozen parent-only probe metadata mismatch")
    probe = ParentOnlyRetentionProbe(dimension, 4, 512)
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    return probe.to(device).eval(), checkpoint


@torch.inference_mode()
def _probe_predictions(probe, values: np.ndarray, device: torch.device) -> np.ndarray:
    outputs = []
    for start in range(0, len(values), 512):
        outputs.append(probe(torch.from_numpy(values[start:start + 512]).to(device)).cpu())
    return torch.cat(outputs).numpy()


def retention_report(calibrator: AffineCalibrator, adapted: np.ndarray, candidate_raw: np.ndarray, baseline_raw: np.ndarray, records: list[dict], probe, device: torch.device, tolerance: float) -> dict:
    indexes = np.asarray([i for i, record in enumerate(records) if record["leaf_count"] == 4])
    restored = calibrator.inverse(adapted[indexes]).astype(np.float32)
    roundtrip = np.sum(
        restored * candidate_raw[indexes] /
        (np.linalg.norm(restored, axis=1, keepdims=True) * np.linalg.norm(candidate_raw[indexes], axis=1, keepdims=True)), axis=1)
    baseline_prediction = _probe_predictions(probe, baseline_raw[indexes], device)
    candidate_prediction = _probe_predictions(probe, restored, device)
    children = torch.stack([records[i]["row"]["leaves"].float() for i in indexes]).numpy()
    labels = np.asarray([records[i]["corpus"] for i in indexes])
    scopes = {"pooled": np.arange(len(indexes))}
    scopes.update({corpus: np.flatnonzero(labels == corpus) for corpus in CORPORA})
    values = {}
    for scope, selected in scopes.items():
        base = cosine_mean(baseline_prediction[selected].reshape(-1, baseline_prediction.shape[-1]), children[selected].reshape(-1, children.shape[-1]))
        candidate = cosine_mean(candidate_prediction[selected].reshape(-1, candidate_prediction.shape[-1]), children[selected].reshape(-1, children.shape[-1]))
        values[scope] = {"rows": len(selected), "frozen_baseline_child_cosine": base, "candidate_inverse_child_cosine": candidate, "delta": candidate - base}
    return {
        "protocol": "frozen_seed_specific_common_parent_only_probe_on_deterministic_four_child_rows",
        "inverse_roundtrip_cosine_mean": float(roundtrip.mean()),
        "inverse_roundtrip_cosine_min": float(roundtrip.min()),
        "candidate_to_frozen_baseline_root_cosine": cosine_mean(restored, baseline_raw[indexes]),
        "scopes": values,
        "noninferiority_tolerance": float(tolerance),
        "noninferior_pooled_and_every_corpus": all(value["delta"] >= -float(tolerance) for value in values.values()),
    }


def _lr(cfg: dict, step: int) -> float:
    ft = cfg["fine_tuning"]
    base, minimum = float(ft["learning_rate"]), float(ft["minimum_learning_rate"])
    warmup, total = int(ft["warmup_steps"]), int(ft["steps"])
    if step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return minimum + .5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def _balanced_training_batch(selected: list[dict], seed: int, step: int) -> list[dict]:
    batch = []
    for corpus in CORPORA:
        for leaves in LEAF_COUNTS:
            cell = [r for r in selected if r["corpus"] == corpus and r["leaf_count"] == leaves]
            ranked = sorted(cell, key=lambda r: hashlib.sha256(f"{seed}|{step}|{corpus}|{leaves}|{r['group']}".encode()).digest())
            batch.append(ranked[0])
    return batch


def focused_fine_tune(model, base_checkpoint: dict, selected: list[dict], validation_records: list[dict], base_fit: dict, cfg: dict, attack_cfg: dict, device: torch.device, seed: int) -> tuple[dict, dict]:
    ft = cfg["fine_tuning"]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ft["learning_rate"]), weight_decay=float(ft["weight_decay"]))
    key_to_index = {(r["corpus"], r["group"], r["leaf_count"]): i for i, r in enumerate(selected)}
    history, selection_values, best = [], [], None
    maximum_training_angle = 0.0
    for step in range(1, int(ft["steps"]) + 1):
        batch = _balanced_training_batch(selected, seed, step)
        optimizer.zero_grad(set_to_none=True)
        prediction, target, weights, health = _merge_records(model, batch, device)
        base_indexes = [key_to_index[(r["corpus"], r["group"], r["leaf_count"])] for r in batch]
        frozen_root = torch.from_numpy(base_fit["roots"][base_indexes]).to(device)
        frozen_weights = torch.from_numpy(base_fit["weights"][base_indexes]).to(device)
        teacher_loss = 1.0 - F.cosine_similarity(prediction, target, dim=1).mean()
        geometry_loss = 1.0 - F.cosine_similarity(prediction, frozen_root, dim=1).mean()
        mean_loss = (prediction.mean(0) - target.mean(0)).square().sum()
        prediction_var = (prediction - prediction.mean(0)).square().mean(0)
        target_var = (target - target.mean(0)).square().mean(0)
        variance_loss = (prediction_var - target_var).square().sum()
        importance_loss = F.mse_loss(weights, frozen_weights)
        loss = (
            float(ft["teacher_weight"]) * teacher_loss
            + float(ft["frozen_geometry_weight"]) * geometry_loss
            + float(ft["distribution_mean_weight"]) * mean_loss
            + float(ft["distribution_variance_weight"]) * variance_loss
            + float(ft["frozen_importance_weight"]) * importance_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite focused loss at step {step}")
        if bool(ft["reject_any_geodesic_saturation"]) and health["geodesic_saturation_count"]:
            raise RuntimeError(f"geodesic saturation at TRAIN step {step}")
        if bool(ft["reject_any_importance_saturation"]) and health["importance_saturation_count"]:
            raise RuntimeError(f"importance saturation at TRAIN step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(ft["gradient_clip"]), error_if_nonfinite=True)
        current_lr = _lr(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        maximum_training_angle = max(maximum_training_angle, float(health["geodesic_angle_max"]))
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "teacher_loss": float(teacher_loss.detach()),
            "frozen_geometry_loss": float(geometry_loss.detach()),
            "distribution_mean_loss": float(mean_loss.detach()),
            "distribution_variance_loss": float(variance_loss.detach()),
            "frozen_importance_loss": float(importance_loss.detach()),
            "gradient_norm": float(grad_norm),
            "learning_rate": current_lr,
            **health,
        }
        if step % int(ft["validation_every"]) == 0:
            validation = extract_direct(model, validation_records, device, int(cfg["root_batch_size"]))
            pooled_attack = fixed_attacks(
                validation["targets"], validation["roots"], validation["groups"], attack_cfg,
                0, int(cfg["bootstrap_seed"]),
            )
            raw = float(pooled_attack["strongest_orientation_safe_auc"])
            selection_values.append(raw)
            smoothed = float(np.mean(selection_values[-int(ft["moving_average_window"]):]))
            pooled_cosine = cosine_mean(validation["roots"], validation["targets"])
            corpus_cosines = {
                corpus: cosine_mean(validation["roots"][validation["corpora"] == corpus], validation["targets"][validation["corpora"] == corpus])
                for corpus in CORPORA
            }
            eligible = (
                pooled_cosine >= float(ft["minimum_validation_pooled_cosine"])
                and min(corpus_cosines.values()) >= float(ft["minimum_validation_per_corpus_cosine"])
                and validation["telemetry"]["geodesic_saturation_count"] == 0
                and validation["telemetry"]["importance_saturation_count"] == 0
                and np.isfinite(validation["roots"]).all()
            )
            record["validation"] = {
                "pooled_strongest_orientation_safe_auc": raw,
                "selection_ma3": smoothed,
                "pooled_teacher_cosine": pooled_cosine,
                "per_corpus_teacher_cosine": corpus_cosines,
                "telemetry": validation["telemetry"],
                "eligible": bool(eligible),
            }
            if eligible and (best is None or smoothed < best["selection_ma3"]):
                best = {
                    "step": step,
                    "selection_raw": raw,
                    "selection_ma3": smoothed,
                    "pooled_teacher_cosine": pooled_cosine,
                    "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                }
        history.append(record)
    if best is None:
        raise RuntimeError("focused run produced no cosine/saturation-eligible checkpoint")
    model.load_state_dict(best["state_dict"], strict=True)
    model.to(device).eval()
    checkpoint = {
        "state_dict": best.pop("state_dict"),
        "variant": "encoder",
        "config": deepcopy(base_checkpoint["config"]),
        "architecture": model.architecture_metadata(),
        "seed": seed,
        "selected_step": best["step"],
        "deployment_fine_tuning": {
            "study": cfg["study"],
            "fit_role": "train",
            "rows": len(selected),
            "balanced_corpus_arity": True,
            "configured_steps": int(ft["steps"]),
            "selection_metric": ft["selection_metric"],
            "selection_moving_average_window": int(ft["moving_average_window"]),
        },
    }
    summary = {
        "executed": True,
        "configured_steps": int(ft["steps"]),
        "selected_step": best["step"],
        "selected_raw_auc": best["selection_raw"],
        "selected_ma3_auc": best["selection_ma3"],
        "selected_pooled_teacher_cosine": best["pooled_teacher_cosine"],
        "maximum_training_geodesic_angle": maximum_training_angle,
        "maximum_training_geodesic_saturation_count": max(r["geodesic_saturation_count"] for r in history),
        "maximum_training_importance_saturation_count": max(r["importance_saturation_count"] for r in history),
        "validation_evaluations": sum("validation" in r for r in history),
        "history": history,
    }
    return checkpoint, summary


def stage_evaluation(model, fit_records: list[dict], validation_records: list[dict], baseline_validation: dict, cfg: dict, attack_cfg: dict, device: torch.device, seed_output: Path, root: Path, label: str, probe, tolerance: float) -> tuple[dict, dict, dict]:
    fit = extract_direct(model, fit_records, device, int(cfg["root_batch_size"]))
    validation = extract_direct(model, validation_records, device, int(cfg["root_batch_size"]))
    fit_groups = np.asarray([f"{r['corpus']}|{r['group']}" for r in fit_records])
    validation_groups = np.asarray([f"{r['corpus']}|{r['group']}" for r in validation_records])
    calibrator = fit_procrustes(fit["roots"], fit["targets"], fit_groups, validation_groups)
    transform = save_transform(calibrator, seed_output / "transforms" / f"{label}_procrustes.npz", root)
    adapted, pre_norm = calibrator.apply_with_norm(validation["roots"])
    scopes = evaluate_scopes(adapted, validation["targets"], validation["groups"], validation["corpora"], pre_norm, cfg, attack_cfg)
    retention = retention_report(
        calibrator, adapted, validation["roots"], baseline_validation["roots"], validation_records,
        probe, device, tolerance,
    )
    inverse = calibrator.inverse(adapted)
    result = {
        "label": label,
        "fit_rows": len(fit_records),
        "fit_source_documents": len(set(fit_groups.tolist())),
        "fit_counts": characterize_records(fit_records),
        "transform": transform,
        "validation": {
            "counts": characterize_records(validation_records),
            "telemetry": validation["telemetry"],
            "scopes": scopes,
        },
        "retention": retention,
        "inverse_roundtrip_cosine": cosine_mean(inverse, validation["roots"]),
    }
    return result, fit, validation


def raw_evaluation(validation: dict, validation_records: list[dict], cfg: dict, attack_cfg: dict) -> dict:
    pre_norm = np.linalg.norm(validation["roots"], axis=1)
    return {
        "counts": characterize_records(validation_records),
        "telemetry": validation["telemetry"],
        "scopes": evaluate_scopes(
            validation["roots"], validation["targets"], validation["groups"], validation["corpora"], pre_norm, cfg, attack_cfg),
    }


def run_seed(root: Path, cfg: dict, seed_cfg: dict, caches: dict, cache_provenance: dict, attack_cfg: dict, output: Path, device: torch.device) -> dict:
    seed = int(seed_cfg["seed"])
    seed_output = output / f"seed_{seed}"
    (seed_output / "transforms").mkdir(parents=True, exist_ok=False)
    checkpoint_path = verify_file(root, seed_cfg["checkpoint"], seed_cfg["checkpoint_sha256"])
    probe_path = verify_file(root, seed_cfg["frozen_probe"], seed_cfg["frozen_probe_sha256"])
    model, base_checkpoint = build_model(checkpoint_path, device)
    if model.arity != 4 or int(base_checkpoint.get("seed", -1)) != seed:
        raise ValueError("checkpoint seed/arity mismatch")
    train_records = eligible_records(caches["train"], cfg, "train")
    validation_records = eligible_records(caches["validation"], cfg, "validation")
    selected = select_balanced_fit(train_records, int(cfg["fit_groups_per_corpus_arity"]), int(cfg["fit_seed"]))
    probe, probe_metadata = load_probe(probe_path, seed_cfg["frozen_probe_sha256"], model.d, device)

    baseline_validation = extract_direct(model, validation_records, device, int(cfg["root_batch_size"]))
    raw = raw_evaluation(baseline_validation, validation_records, cfg, attack_cfg)
    simple, base_fit, _ = stage_evaluation(
        model, selected, validation_records, baseline_validation, cfg, attack_cfg, device,
        seed_output, root, "simple", probe, float(cfg["retention"]["cosine_noninferiority_tolerance"]),
    )
    simple_pooled = simple["validation"]["scopes"]["pooled"]
    misses_margin = (
        simple_pooled["attacks"]["strongest_orientation_safe_auc"] > float(cfg["margin"]["maximum_pooled_strongest_auc"])
        or simple_pooled["teacher_cosine"] < float(cfg["margin"]["minimum_pooled_teacher_cosine"])
    )
    fine_summary = {"executed": False, "reason": "simple_adapter_met_predeclared_margin"}
    final = simple
    final_checkpoint = None
    if misses_margin and bool(cfg["fine_tuning"]["enabled_if_simple_adapter_misses_margin"]):
        tuned_checkpoint, fine_summary = focused_fine_tune(
            model, base_checkpoint, selected, validation_records, base_fit, cfg, attack_cfg, device, seed,
        )
        checkpoint_out = seed_output / f"focused_encoder_seed{seed}.pt"
        torch.save(tuned_checkpoint, checkpoint_out)
        final_checkpoint = {
            "path": str(checkpoint_out.relative_to(root)),
            "sha256": sha256_file(checkpoint_out),
            "selected_step": tuned_checkpoint["selected_step"],
        }
        final, _, _ = stage_evaluation(
            model, selected, validation_records, baseline_validation, cfg, attack_cfg, device,
            seed_output, root, "post_fine_tune", probe,
            float(cfg["retention"]["cosine_noninferiority_tolerance"]),
        )
    final_pooled = final["validation"]["scopes"]["pooled"]
    screen_passed = (
        final_pooled["attacks"]["strongest_orientation_safe_auc"] <= float(cfg["screen"]["maximum_pooled_strongest_auc"])
        and final_pooled["teacher_cosine"] > float(cfg["screen"]["minimum_pooled_teacher_cosine_exclusive"])
    )
    result = {
        "seed": seed,
        "checkpoint": seed_cfg["checkpoint"],
        "checkpoint_sha256": seed_cfg["checkpoint_sha256"],
        "frozen_probe": seed_cfg["frozen_probe"],
        "frozen_probe_sha256": seed_cfg["frozen_probe_sha256"],
        "frozen_probe_updates": probe_metadata.get("updates"),
        "cache_provenance": cache_provenance,
        "eligible_population": {"train": characterize_records(train_records), "validation": characterize_records(validation_records)},
        "balanced_fit": characterize_records(selected),
        "raw_current_k4": raw,
        "simple_adapter": simple,
        "simple_missed_margin": bool(misses_margin),
        "fine_tuning": fine_summary,
        "fine_tuned_checkpoint": final_checkpoint,
        "final_stage": final["label"],
        "final": final,
        "screen_passed": bool(screen_passed),
    }
    (seed_output / "study.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _attack_rows(seed_result: dict) -> list[dict]:
    rows = []
    stages = {"raw": seed_result["raw_current_k4"], "simple": seed_result["simple_adapter"]["validation"]}
    if seed_result["final_stage"] != "simple":
        stages["final"] = seed_result["final"]["validation"]
    for stage, entry in stages.items():
        for scope, values in entry["scopes"].items():
            for attack, metrics in values["attacks"]["attacks"].items():
                rows.append({
                    "seed": seed_result["seed"], "stage": stage, "scope": scope, "attack": attack,
                    "rows": values["rows"], "source_documents": values["source_documents"],
                    "teacher_cosine": values["teacher_cosine"],
                    "raw_auc": metrics["raw_auc"], "orientation_safe_auc": metrics["orientation_safe_auc"],
                    "auc_ci95_low": None if metrics["orientation_safe_auc_ci95_group_bootstrap"] is None else metrics["orientation_safe_auc_ci95_group_bootstrap"][0],
                    "auc_ci95_high": None if metrics["orientation_safe_auc_ci95_group_bootstrap"] is None else metrics["orientation_safe_auc_ci95_group_bootstrap"][1],
                    "pr_auc": metrics["pr_auc"], "balanced_accuracy": metrics["balanced_accuracy"],
                })
    return rows


def render_report(result: dict) -> str:
    lines = [
        "# K4 deterministic one-level deployment-regime improvement study", "",
        "**Status:** TRAIN/validation-only development report. This is not an official success claim and does not authorize a new protocol or data stage.", "",
        "Population: deterministic fixed-prefix one-level roots, 2-4 children, 1025..4096 tokens. Attacks use source-document grouped splits.", "",
        "## Seed summary", "",
        "| seed | final stage | pooled logistic safe AUC | pooled RBF safe AUC | pooled strongest | cosine | strongest CI95 | rank | condition | inverse min | retention worst delta | screen |",
        "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for seed in result["seeds"]:
        final = seed["final"]
        pooled = final["validation"]["scopes"]["pooled"]
        attacks = pooled["attacks"]["attacks"]
        strongest_ci = pooled["attacks"]["strongest_orientation_safe_auc_ci95"]
        retention = final["retention"]
        lines.append(
            f"| {seed['seed']} | {seed['final_stage']} | {attacks['standard_scaled_logistic']['orientation_safe_auc']:.6f} | "
            f"{attacks['standard_scaled_rbf_svm']['orientation_safe_auc']:.6f} | {pooled['attacks']['strongest_orientation_safe_auc']:.6f} | "
            f"{pooled['teacher_cosine']:.6f} | {strongest_ci} | {final['transform']['rank']}/{final['transform']['dimension']} | "
            f"{final['transform']['condition']:.6g} | {retention['inverse_roundtrip_cosine_min']:.9f} | "
            f"{min(v['delta'] for v in retention['scopes'].values()):+.3e} | {seed['screen_passed']} |"
        )
    lines += ["", "## Eligible deterministic validation counts", ""]
    if result["seeds"]:
        counts = result["seeds"][0]["eligible_population"]["validation"]
        lines += ["| corpus | 2-child | 3-child | 4-child | total source docs |", "|---|---:|---:|---:|---:|"]
        for corpus in CORPORA:
            cell = counts["by_corpus_arity"][corpus]
            lines.append(f"| {corpus} | {cell['2']['rows']} | {cell['3']['rows']} | {cell['4']['rows']} | {sum(cell[str(k)]['source_documents'] for k in LEAF_COUNTS)} |")
        lines.append(f"\nPooled rows/source documents: **{counts['rows']} / {counts['source_documents']}**. Balanced Procrustes/fine-tuning fit: 120 unique TRAIN documents in each of 12 corpus x child-count cells (1,440 rows).")
    lines += [
        "", "## Current K4 and simple TRAIN Procrustes characterization", "",
        "| seed | stage | scope | logistic safe AUC | RBF safe AUC | strongest | cosine | effective rank | collapse |",
        "|---:|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in result["seeds"]:
        stages = {
            "raw K4": seed["raw_current_k4"],
            "simple Procrustes": seed["simple_adapter"]["validation"],
        }
        for stage, entry in stages.items():
            for scope, values in entry["scopes"].items():
                attacks = values["attacks"]["attacks"]
                lines.append(
                    f"| {seed['seed']} | {stage} | {scope} | "
                    f"{attacks['standard_scaled_logistic']['orientation_safe_auc']:.6f} | "
                    f"{attacks['standard_scaled_rbf_svm']['orientation_safe_auc']:.6f} | "
                    f"{values['attacks']['strongest_orientation_safe_auc']:.6f} | "
                    f"{values['teacher_cosine']:.6f} | {values['diagnostics']['effective_rank']:.3f} | "
                    f"{values['diagnostics']['collapse']} |"
                )
    lines += ["", "## Final fixed attacks by corpus", "", "| seed | scope | logistic safe AUC (CI95) | RBF safe AUC (CI95) | strongest | cosine (CI95) | rows |", "|---:|---|---:|---:|---:|---:|---:|"]
    for seed in result["seeds"]:
        for scope, values in seed["final"]["validation"]["scopes"].items():
            attacks = values["attacks"]["attacks"]
            logistic = attacks["standard_scaled_logistic"]
            rbf = attacks["standard_scaled_rbf_svm"]
            lines.append(
                f"| {seed['seed']} | {scope} | {logistic['orientation_safe_auc']:.6f} {logistic['orientation_safe_auc_ci95_group_bootstrap']} | "
                f"{rbf['orientation_safe_auc']:.6f} {rbf['orientation_safe_auc_ci95_group_bootstrap']} | "
                f"{values['attacks']['strongest_orientation_safe_auc']:.6f} | {values['teacher_cosine']:.6f} {values['teacher_cosine_ci95_row_bootstrap']} | {values['rows']} |"
            )
    lines += ["", "## Fine-tuning and health", ""]
    for seed in result["seeds"]:
        fine = seed["fine_tuning"]
        final = seed["final"]
        telemetry = final["validation"]["telemetry"]
        pooled_health = final["validation"]["scopes"]["pooled"]["diagnostics"]
        lines.append(
            f"- Seed {seed['seed']}: simple margin miss={seed['simple_missed_margin']}; focused executed={fine['executed']}; "
            f"selected step={fine.get('selected_step')}; validation evaluations={fine.get('validation_evaluations', 0)}; "
            f"geodesic/importance saturation={telemetry['geodesic_saturation_count']}/{telemetry['importance_saturation_count']}; "
            f"effective rank={pooled_health['effective_rank']:.3f}; unique fraction={pooled_health['unique_rounded_fraction']:.6f}; "
            f"collapse={pooled_health['collapse']}."
        )
    lines += ["", "## Retention and provenance", ""]
    for seed in result["seeds"]:
        final = seed["final"]
        retention = final["retention"]
        lines.append(
            f"- Seed {seed['seed']}: transform rank {final['transform']['rank']}/{final['transform']['dimension']}, "
            f"singular range [{final['transform']['minimum_singular_value']:.9f}, {final['transform']['maximum_singular_value']:.9f}], "
            f"inverse min {retention['inverse_roundtrip_cosine_min']:.9f}, frozen-probe noninferior={retention['noninferior_pooled_and_every_corpus']}, "
            f"checkpoint/probe hashes `{seed['checkpoint_sha256']}` / `{seed['frozen_probe_sha256']}`."
        )
    lines += [
        "", f"Seed-424242 screen allowed additional seeds: **{result['additional_seeds_authorized_by_screen']}**.",
        "", "Every opened cache tensor/sidecar, checkpoint, probe, transform, config, and attack config hash is recorded in the JSON artifacts. Source sidecars were rejected unless their complete half-open ranges ended below 20000. No future deployment outcome was used for fitting or selection.",
        "", "Do not interpret this validation study as official success.",
    ]
    return "\n".join(lines) + "\n"


def run_study(root: Path, config_path: Path, device_name: str) -> dict:
    started = time.time()
    reject_path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    validate_config(cfg)
    output = root / cfg["output_dir"]
    reject_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite study output: {output}")
    output.mkdir(parents=True)
    attack_path = verify_file(root, cfg["attack_config"], cfg["attack_config_sha256"])
    attack_cfg = yaml.safe_load(attack_path.read_text())
    caches, provenance = load_caches(root, cfg)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    seed_results = []
    first = run_seed(root, cfg, cfg["seeds"][0], caches, provenance, attack_cfg, output, device)
    seed_results.append(first)
    additional = bool(first["screen_passed"])
    if additional:
        for seed_cfg in cfg["seeds"][1:]:
            seed_results.append(run_seed(root, cfg, seed_cfg, caches, provenance, attack_cfg, output, device))
    result = {
        "study": cfg["study"],
        "validation_only": True,
        "roles_opened": list(ROLES),
        "source_index_ceiling_exclusive": int(cfg["maximum_source_index_exclusive"]),
        "future_deployment_data_opened": False,
        "official_success_declared": False,
        "config": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "attack_config": cfg["attack_config"],
        "attack_config_sha256": sha256_file(attack_path),
        "seeds": seed_results,
        "additional_seeds_authorized_by_screen": additional,
        "runtime_seconds": time.time() - started,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
    }
    (output / "study.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = [row for seed in seed_results for row in _attack_rows(seed)]
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "REPORT.md").write_text(render_report(result))
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/kary4_deployment_regime_validation_v1/config.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = run_study(root, root / args.config, args.device)
    print(json.dumps({
        "study": result["study"],
        "seeds_run": [seed["seed"] for seed in result["seeds"]],
        "additional_seeds_authorized_by_screen": result["additional_seeds_authorized_by_screen"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
