"""Fail-closed, evaluator-only V3 CLS two-arm protocol.

This module deliberately has no dataset or teacher imports.  It consumes an
already materialized, hash-addressed cache.  The safe arm is teacher-comparable;
the unrestricted full-document arm never loads or reports a root teacher target.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .architecture import GeodesicMerge
from .deployment_regime_study import fixed_attacks
from .evaluate_kary_validation import ParentOnlyRetentionProbe
from .train import _enforce_v3_cls_guards
from .validation_output_calibration import diagnostics

TEACHER_ID = "Alibaba-NLP/gte-modernbert-base"
TEACHER_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")
SOURCE_RANGES = {"development": (308000, 317000), "official": (318000, 348000)}
ARITY = 4
CHUNK_TOKENS = 1024
SAFE_RANGE = (1025, 4096)
FORBIDDEN_FULL_KEYS = frozenset({"teacher_target", "root_target", "targets", "teacher_root"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_namespace(path: str | Path, label: str) -> None:
    text = str(path).lower()
    if "v3_cls" not in text:
        raise ValueError(f"{label} must be isolated in a v3_cls namespace: {path}")
    if any(token in text for token in ("v1", "v2_cls", "v2_2_cls", "kary4_strict_autoencoder_v6")):
        raise ValueError(f"{label} contains a forbidden historical namespace: {path}")


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if not expected or actual != str(expected).lower():
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} actual={actual}")


def validate_protocol(cfg: dict) -> None:
    """Validate immutable evaluator semantics without opening any data artifact."""
    if cfg.get("stage") != "v3_cls_two_arm_evaluator" or int(cfg.get("protocol_version", -1)) != 1:
        raise ValueError("wrong V3 evaluator stage/protocol")
    if cfg.get("v3_cls_experiment") is not True:
        raise ValueError("V3 evaluator must declare v3_cls_experiment")
    if (cfg.get("model_name"), cfg.get("model_revision"), cfg.get("teacher_pooling")) != (
        TEACHER_ID, TEACHER_REVISION, "cls"
    ):
        raise ValueError("V3 evaluator requires the pinned normalized CLS teacher")
    if (int(cfg.get("arity", -1)), int(cfg.get("chunk_tokens", -1)), cfg.get("reduction")) != (
        ARITY, CHUNK_TOKENS, "deterministic_contiguous_k4_carry_singleton"
    ):
        raise ValueError("V3 evaluator requires exact contiguous K=4 recursion")
    if cfg.get("safe_arm") != {
        "minimum_tokens": 1025,
        "maximum_tokens": 4096,
        "direct_single_merge": True,
        "teacher_target": "normalized_cls_of_exact_safe_prefix",
    }:
        raise ValueError("safe arm contract changed")
    if cfg.get("full_arm") != {
        "every_available_token": True,
        "leaf_tokens": 1024,
        "retain_final_short_leaf": True,
        "token_cap": None,
        "leaf_cap": None,
        "depth_cap": None,
        "truncation": False,
        "teacher_target": False,
        "teacher_metrics": False,
        "attacks": False,
    }:
        raise ValueError("full arm acquired a cap, truncation, teacher metric, or attack")
    if tuple(cfg.get("corpora", ())) != CORPORA:
        raise ValueError("V3 corpus order changed")
    if cfg.get("source_ranges") != {key: list(value) for key, value in SOURCE_RANGES.items()}:
        raise ValueError("V3 evaluation source allocation changed")
    if cfg.get("attack_config") != "configs/evaluator_attacks_v1.yaml":
        raise ValueError("fixed attack configuration changed")
    if int(cfg.get("bootstrap_replicates", -1)) != 400:
        raise ValueError("V3 requires 400 bootstrap replicates")
    if not isinstance(cfg.get("bootstrap_seed"), int):
        raise ValueError("V3 bootstrap seed must be fixed")
    collapse = cfg.get("collapse", {})
    for key in ("minimum_effective_rank", "minimum_unique_rounded_fraction", "minimum_pre_normalization_norm"):
        if key not in collapse:
            raise ValueError(f"missing collapse guard: {key}")


def _validate_teacher_provenance(value: dict, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"missing {label} teacher provenance")
    model = value.get("teacher_id", value.get("model", value.get("model_name")))
    revision = value.get("teacher_revision", value.get("revision", value.get("model_revision")))
    if (model, revision, value.get("teacher_pooling", value.get("pooling"))) != (
        TEACHER_ID, TEACHER_REVISION, "cls"
    ):
        raise ValueError(f"{label} is not pinned V3 CLS provenance")


def load_checkpoint(root: Path, spec: dict, device: torch.device) -> tuple[GeodesicMerge, dict, dict]:
    """Load only a hash-pinned, fresh V3 checkpoint and its TRAIN cache manifest."""
    checkpoint_path = root / spec["checkpoint"]
    training_manifest_path = root / spec["training_cache_manifest"]
    _require_namespace(checkpoint_path, "checkpoint")
    _require_namespace(training_manifest_path, "training cache")
    _require_hash(checkpoint_path, spec.get("checkpoint_sha256", ""), "V3 checkpoint")
    _require_hash(training_manifest_path, spec.get("training_cache_manifest_sha256", ""), "V3 TRAIN cache manifest")
    training_manifest = json.loads(training_manifest_path.read_text())
    if training_manifest.get("study") != "v3_cls" or training_manifest.get("role") != "train":
        raise ValueError("checkpoint TRAIN cache manifest has the wrong study/role")
    if training_manifest.get("source_range") != [296000, 308000]:
        raise ValueError("checkpoint TRAIN cache range changed")
    _validate_teacher_provenance(training_manifest.get("teacher_provenance", {}), "TRAIN cache")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("config", {})
    if cfg.get("v3_cls_experiment") is not True:
        raise ValueError("checkpoint does not declare the V3 CLS experiment")
    _enforce_v3_cls_guards(cfg)
    if cfg.get("teacher_trust_remote_code") is not True:
        raise ValueError("checkpoint lacks the required pinned-teacher execution provenance")
    if any(cfg.get(key) not in (None, False) for key in (
        "initialize_encoder_from_autoencoder_checkpoint", "initialize_from_encoder_checkpoint",
        "encoder_initialization_checkpoint", "continue_encoder_from_checkpoint",
        "continue_autoencoder_from_checkpoint",
    )):
        raise ValueError("checkpoint violates fresh random V3 initialization")
    if (cfg.get("decoder_mode"), cfg.get("decoder_length_mode"), cfg.get("geodesic_parameterization")) != (
        "strict", "predicted", "tangent_expmap"
    ):
        raise ValueError("checkpoint is not the frozen V3 strict-decoder architecture")
    if int(checkpoint.get("seed", -1)) != int(spec["seed"]):
        raise ValueError("checkpoint seed differs from evaluation specification")
    if checkpoint.get("variant") != "autoencoder" or int(cfg.get("arity", -1)) != ARITY:
        raise ValueError("V3 evaluator requires the strict K4 autoencoder checkpoint")
    if cfg.get("checkpoint_selection") != "fixed_final_train_only" or int(checkpoint.get("selected_step", -1)) != int(cfg.get("steps", -2)):
        raise ValueError("checkpoint is not the fixed final TRAIN-only checkpoint")
    configured_cache = Path(str(cfg.get("cache_dir", "")))
    if configured_cache.name != training_manifest_path.parent.name:
        raise ValueError("checkpoint cache and pinned TRAIN manifest disagree")
    state = checkpoint.get("state_dict", {})
    if "input.0.weight" not in state:
        raise ValueError("checkpoint has no merger state")
    dimension = int(state["input.0.weight"].shape[1] - 2)
    model = GeodesicMerge(
        d=dimension, hidden=int(cfg["encoder_hidden"]), layers=int(cfg["encoder_layers"]),
        heads=int(cfg["heads"]), max_angle=float(cfg["max_geodesic_angle"]), decoder=True,
        decoder_layers=int(cfg["decoder_layers"]), decoder_length_mode=str(cfg["decoder_length_mode"]),
        importance_mode=str(cfg["importance_mode"]), importance_beta=float(cfg["importance_beta"]),
        decoder_mode=str(cfg["decoder_mode"]), geodesic_parameterization=str(cfg["geodesic_parameterization"]),
        arity=ARITY,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    provenance = {
        "checkpoint": spec["checkpoint"], "checkpoint_sha256": spec["checkpoint_sha256"],
        "training_cache_manifest": spec["training_cache_manifest"],
        "training_cache_manifest_sha256": spec["training_cache_manifest_sha256"],
        "seed": int(spec["seed"]), "fresh_v3_cls": True,
    }
    return model, checkpoint, provenance


def _artifact_path_and_hash(root: Path, cache_root: Path, manifest: dict, relative: str) -> Path:
    entry = manifest.get("artifacts", {}).get(relative)
    expected = entry.get("sha256") if isinstance(entry, dict) else entry
    if not expected:
        raise ValueError(f"cache manifest does not hash {relative}")
    path = cache_root / relative
    try:
        path.resolve().relative_to(cache_root.resolve())
    except ValueError as exc:
        raise ValueError("cache artifact escaped V3 cache directory") from exc
    _require_hash(path, expected, f"evaluation cache artifact {relative}")
    return path


def _validate_safe_row(row: dict, role: str, source_range: tuple[int, int]) -> None:
    required = ("id", "source_index", "leaves", "lengths", "teacher_target")
    if any(key not in row for key in required):
        raise ValueError("safe row is incomplete")
    count = int(row["leaves"].shape[0])
    lengths = row["lengths"]
    tokens = int(lengths.sum().item())
    if count not in (2, 3, 4) or row["leaves"].ndim != 2 or lengths.shape != (count,):
        raise ValueError("safe arm must be one direct 2..4-child merge")
    if not SAFE_RANGE[0] <= tokens <= SAFE_RANGE[1] or torch.any(lengths <= 0) or torch.any(lengths > CHUNK_TOKENS):
        raise ValueError("safe arm escaped 1025..4096 exact leaf coverage")
    if count > 1 and not torch.all(lengths[:-1] == CHUNK_TOKENS):
        raise ValueError("safe arm has a non-final short leaf")
    if not source_range[0] <= int(row["source_index"]) < source_range[1]:
        raise ValueError(f"safe row escaped V3 {role} range")
    if row.get("teacher_pooling", "cls") != "cls":
        raise ValueError("safe target is not declared CLS")
    if not torch.isfinite(row["leaves"]).all() or not torch.isfinite(lengths).all() or not torch.isfinite(row["teacher_target"]).all():
        raise FloatingPointError("safe cache contains non-finite tensors")


def _validate_full_shard(shard: dict, item: dict, source_range: tuple[int, int]) -> None:
    present = FORBIDDEN_FULL_KEYS.intersection(shard)
    if present:
        raise ValueError(f"full arm contains forbidden teacher target fields: {sorted(present)}")
    if shard.get("id") != item.get("id") or int(shard.get("source_index", -1)) != int(item.get("source_index", -2)):
        raise ValueError("safe/full source identity mismatch")
    if not source_range[0] <= int(shard["source_index"]) < source_range[1]:
        raise ValueError("full shard escaped V3 evaluation range")
    if shard.get("truncated") is not False or shard.get("token_cap") is not None or shard.get("leaf_cap") is not None:
        raise ValueError("full document was truncated or capped")
    leaves, lengths = shard.get("leaves"), shard.get("lengths")
    if not isinstance(leaves, torch.Tensor) or not isinstance(lengths, torch.Tensor) or leaves.ndim != 2 or lengths.shape != (len(leaves),) or len(leaves) < 2:
        raise ValueError("invalid full-document leaves")
    if torch.any(lengths <= 0) or torch.any(lengths > CHUNK_TOKENS) or (len(lengths) > 1 and not torch.all(lengths[:-1] == CHUNK_TOKENS)):
        raise ValueError("full-document leaves are not exact contiguous 1024-token chunks")
    tokens = int(lengths.sum().item())
    if tokens != int(shard.get("token_count", -1)) or tokens != int(item.get("token_count", -2)):
        raise ValueError("full-document token accounting mismatch")
    if not torch.isfinite(leaves).all() or not torch.isfinite(lengths).all():
        raise FloatingPointError("full cache contains non-finite tensors")


def load_two_arm_cache(root: Path, spec: dict, protocol: dict) -> tuple[dict, dict, dict]:
    """Open a complete hash-pinned V3 cache; this is the only evaluator data entrypoint."""
    role = str(spec.get("role"))
    if role not in SOURCE_RANGES:
        raise PermissionError("V3 evaluator accepts only allocated development or official roles")
    manifest_path = root / spec["cache_manifest"]
    _require_namespace(manifest_path, "evaluation cache")
    _require_hash(manifest_path, spec.get("cache_manifest_sha256", ""), "evaluation cache manifest")
    cache_root = manifest_path.parent
    complete = cache_root / "COMPLETE"
    if not complete.is_file() or complete.read_text().strip() != f"{sha256_file(manifest_path)}  {manifest_path.name}":
        raise RuntimeError("evaluation cache is not atomically complete")
    manifest = json.loads(manifest_path.read_text())
    expected_range = SOURCE_RANGES[role]
    if manifest.get("schema") != "v3_cls_two_arm_cache_v1" or manifest.get("study") != "v3_cls" or manifest.get("role") != role:
        raise ValueError("wrong V3 two-arm cache schema/study/role")
    if manifest.get("source_range") != list(expected_range):
        raise ValueError("evaluation cache source range changed")
    _validate_teacher_provenance(manifest.get("teacher_provenance", {}), "evaluation cache")
    if manifest.get("safe_arm") != protocol["safe_arm"] or manifest.get("full_arm") != protocol["full_arm"]:
        raise ValueError("cache arm contract differs from evaluator protocol")

    safe_by, full_by = {}, {}
    for corpus in CORPORA:
        safe_path = _artifact_path_and_hash(root, cache_root, manifest, f"{corpus}/safe.pt")
        index_path = _artifact_path_and_hash(root, cache_root, manifest, f"{corpus}/full_index.json")
        safe = torch.load(safe_path, map_location="cpu", weights_only=False)
        index = json.loads(index_path.read_text()).get("documents", [])
        if not safe or len(safe) != len(index):
            raise ValueError(f"safe/full count mismatch for {corpus}")
        for row, item in zip(safe, index):
            _validate_safe_row(row, role, expected_range)
            if str(row["id"]) != str(item.get("id")) or int(row["source_index"]) != int(item.get("source_index", -1)):
                raise ValueError("safe and full arms do not use identical source order")
            # Hash every shard up front, but keep full documents lazy so the
            # unrestricted arm cannot exhaust RAM merely by having no leaf cap.
            shard_path = _artifact_path_and_hash(root, cache_root, manifest, str(item["shard"]))
            item["_shard_path"] = str(shard_path)
            item["_source_range"] = expected_range
        safe_by[corpus], full_by[corpus] = safe, index
    ids = [f"{c}|{row['id']}" for c in CORPORA for row in safe_by[c]]
    provenance = {
        "manifest": spec["cache_manifest"], "manifest_sha256": spec["cache_manifest_sha256"],
        "role": role, "source_range": list(expected_range), "document_order_sha256": stable_hash(ids),
        "teacher_pooling": "cls", "same_documents_both_arms": True,
    }
    return safe_by, full_by, provenance


class _Stats:
    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []

    def add(self, value: torch.Tensor) -> None:
        self.parts.append(value.detach().float().cpu().numpy().reshape(-1))

    def result(self) -> dict:
        values = np.concatenate(self.parts) if self.parts else np.asarray([], dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise FloatingPointError("missing/non-finite evaluator telemetry")
        return {"count": int(len(values)), "minimum": float(values.min()), "mean": float(values.mean()), "maximum": float(values.max())}


def _collect_aux(model, telemetry: dict) -> None:
    aux = model._last_merge_aux
    active = aux["active_mask"]
    telemetry["angles"].add(aux["angle"])
    telemetry["corrections"].add(aux["corrections"][active].abs())
    telemetry["geodesic_saturation_count"] += int(aux["geodesic_saturated"].sum())
    telemetry["importance_saturation_count"] += int(((aux["corrections"].abs() >= .95 * model.importance_beta) & active).sum())
    telemetry["importance_active_slots"] += int(active.sum())
    telemetry["merge_calls"] += int(aux["angle"].numel())


def _new_telemetry() -> dict:
    return {"angles": _Stats(), "corrections": _Stats(), "geodesic_saturation_count": 0,
            "importance_saturation_count": 0, "importance_active_slots": 0, "merge_calls": 0}


def _finish_telemetry(value: dict, *, documents: int, leaves: int, depths: list[int], seconds: float) -> dict:
    if seconds <= 0 or not np.isfinite(seconds):
        raise FloatingPointError("invalid evaluator timing")
    return {"documents": documents, "leaves": leaves,
            "depth": {"minimum": min(depths), "mean": float(np.mean(depths)), "maximum": max(depths)},
            "geodesic_angle_radians": value["angles"].result(),
            "importance_correction_absolute": value["corrections"].result(),
            "geodesic_saturation_count": value["geodesic_saturation_count"],
            "importance_saturation_count": value["importance_saturation_count"],
            "importance_active_slots": value["importance_active_slots"], "internal_merges": value["merge_calls"],
            "seconds": seconds, "documents_per_second": documents / seconds, "leaves_per_second": leaves / seconds}


@torch.inference_mode()
def infer_safe(model, safe_by: dict, device: torch.device, batch_size: int) -> dict:
    records = [(corpus, row) for corpus in CORPORA for row in safe_by[corpus]]
    grouped = defaultdict(list)
    for index, (_, row) in enumerate(records):
        grouped[len(row["leaves"])].append((index, row))
    output: list[np.ndarray | None] = [None] * len(records)
    telemetry = _new_telemetry(); dtype = next(model.parameters()).dtype; started = time.perf_counter()
    for count in (2, 3, 4):
        for start in range(0, len(grouped[count]), batch_size):
            part = grouped[count][start:start + batch_size]
            children = torch.stack([row["leaves"].float() for _, row in part]).to(device=device, dtype=dtype)
            masses = torch.stack([row["lengths"].float() for _, row in part]).to(device=device, dtype=dtype)
            if count < ARITY:
                children = F.pad(children, (0, 0, 0, ARITY-count)); masses = F.pad(masses, (0, ARITY-count))
            roots = model.merge(children, masses); _collect_aux(model, telemetry)
            if not torch.isfinite(roots).all(): raise FloatingPointError("non-finite safe roots")
            for (index, _), value in zip(part, roots.float().cpu().numpy()): output[index] = value
    elapsed = time.perf_counter() - started
    roots = np.stack(output).astype(np.float32)
    targets = np.stack([row["teacher_target"].float().numpy() for _, row in records]).astype(np.float32)
    if not np.isfinite(roots).all() or not np.isfinite(targets).all(): raise FloatingPointError("non-finite safe arrays")
    return {"roots": roots, "targets": targets, "groups": np.asarray([f"{c}|{r['id']}" for c, r in records]),
            "corpora": np.asarray([c for c, _ in records]), "records": records,
            "telemetry": _finish_telemetry(telemetry, documents=len(records), leaves=sum(len(r["leaves"]) for _, r in records), depths=[1]*len(records), seconds=elapsed)}


def _depth(leaves: int) -> int:
    depth = 0
    while leaves > 1:
        leaves = (leaves + ARITY - 1) // ARITY; depth += 1
    return depth


@torch.inference_mode()
def reduce_full(model, shard: dict, device: torch.device, telemetry: dict, batch_size: int) -> torch.Tensor:
    nodes, masses = shard["leaves"].float(), shard["lengths"].float()
    dtype = next(model.parameters()).dtype
    while len(nodes) > 1:
        groups = [(start, min(ARITY, len(nodes)-start)) for start in range(0, len(nodes), ARITY)]
        merge_groups = [(i, start, width) for i, (start, width) in enumerate(groups) if width > 1]
        parents = {}
        for offset in range(0, len(merge_groups), batch_size):
            part = merge_groups[offset:offset + batch_size]
            children = torch.stack([F.pad(nodes[s:s+w], (0, 0, 0, ARITY-w)) for _, s, w in part]).to(device=device, dtype=dtype)
            weights = torch.stack([F.pad(masses[s:s+w], (0, ARITY-w)) for _, s, w in part]).to(device=device, dtype=dtype)
            values = model.merge(children, weights); _collect_aux(model, telemetry)
            if not torch.isfinite(values).all(): raise FloatingPointError("non-finite full recursion")
            for local, (index, start, width) in enumerate(part): parents[index] = (values[local].float().cpu(), masses[start:start+width].sum())
        next_nodes, next_masses = [], []
        for index, (start, width) in enumerate(groups):
            value = (nodes[start], masses[start]) if width == 1 else parents[index]
            next_nodes.append(value[0]); next_masses.append(value[1])
        nodes, masses = torch.stack(next_nodes), torch.stack(next_masses)
    return nodes[0]


@torch.inference_mode()
def infer_full(model, full_by: dict, device: torch.device, batch_size: int) -> dict:
    telemetry = _new_telemetry(); roots=[]; labels=[]; depths=[]; leaves=0; started=time.perf_counter()
    for corpus in CORPORA:
        for item in full_by[corpus]:
            shard = item.get("_loaded_shard")
            if shard is None:
                shard = torch.load(item["_shard_path"], map_location="cpu", weights_only=False)
            if "_source_range" not in item:
                raise ValueError("full shard is missing its V3 role range provenance")
            _validate_full_shard(shard, item, tuple(item["_source_range"]))
            count=len(shard["leaves"])
            roots.append(reduce_full(model, shard, device, telemetry, batch_size).numpy()); labels.append(corpus); depths.append(_depth(count)); leaves += count
    elapsed=time.perf_counter()-started; values=np.stack(roots).astype(np.float32)
    if not np.isfinite(values).all(): raise FloatingPointError("non-finite full roots")
    return {"roots":values, "corpora":np.asarray(labels),
            "telemetry":_finish_telemetry(telemetry,documents=len(values),leaves=leaves,depths=depths,seconds=elapsed)}


def _health(values: np.ndarray, cfg: dict) -> dict:
    if not np.isfinite(values).all(): raise FloatingPointError("non-finite roots before rank check")
    result=diagnostics(values,np.linalg.norm(values,axis=1),cfg)
    if result["finite_fraction"] != 1.0 or not np.isfinite(result["effective_rank"]):
        raise FloatingPointError("finite/rank diagnostics failed")
    return result


def _cosine_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    denominator=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    values=np.sum(x*y,axis=1)/denominator
    if not np.isfinite(values).all(): raise FloatingPointError("non-finite safe teacher cosine")
    return values


@torch.inference_mode()
def evaluate_exact4_retention(model, data: dict, spec: dict, root: Path, device: torch.device) -> dict:
    """Evaluate parent-only reconstruction only where four child slots exist."""
    indexes = [i for i, (_, row) in enumerate(data["records"]) if len(row["leaves"]) == ARITY]
    if not indexes:
        raise ValueError("no exact-K4 rows for decoder/probe retention")
    parents = torch.from_numpy(data["roots"][indexes]).float()
    children = torch.stack([data["records"][i][1]["leaves"].float() for i in indexes])
    lengths = torch.stack([data["records"][i][1]["lengths"].float() for i in indexes])
    labels = np.asarray([data["records"][i][0] for i in indexes])
    decoded, importance_parts = [], []
    for start in range(0, len(parents), 512):
        prediction, aux = model.decode(parents[start:start+512].to(device), return_aux=True)
        decoded.append(prediction.float().cpu()); importance_parts.append(aux["importance_weights"].float().cpu())
    decoded = torch.cat(decoded); importance = torch.cat(importance_parts)
    probe_path = root / spec["probe"]
    _require_namespace(probe_path, "parent-only probe")
    _require_hash(probe_path, spec.get("probe_sha256", ""), "TRAIN-only parent probe")
    checkpoint = torch.load(probe_path, map_location="cpu", weights_only=True)
    if checkpoint.get("fit_role") != "v3_cls_train" or int(checkpoint.get("seed", -1)) != int(spec["seed"]) or int(checkpoint.get("arity", -1)) != ARITY:
        raise ValueError("probe is not the matching V3 TRAIN-only exact-K4 probe")
    probe = ParentOnlyRetentionProbe(768, ARITY, int(checkpoint.get("hidden", 512))).to(device)
    probe.load_state_dict(checkpoint["state_dict"], strict=True); probe.eval()
    predicted = torch.cat([probe(parents[i:i+512].to(device)).cpu() for i in range(0, len(parents), 512)])
    def report(prediction: torch.Tensor) -> dict:
        per_doc = F.cosine_similarity(prediction, children, dim=-1).mean(-1).numpy()
        mse = ((prediction - children) ** 2).mean((1, 2)).numpy()
        return {"pooled_ordered_child_cosine": float(per_doc.mean()), "pooled_mse": float(mse.mean()),
                "per_corpus_ordered_child_cosine": {c: float(per_doc[labels == c].mean()) for c in CORPORA}}
    floors = spec.get("probe_floors")
    if not isinstance(floors, dict) or any(k not in floors for k in ("pooled", *CORPORA)):
        raise ValueError("evaluation specification lacks frozen TRAIN-only probe floors")
    probe_report = report(predicted)
    observed = {"pooled": probe_report["pooled_ordered_child_cosine"], **probe_report["per_corpus_ordered_child_cosine"]}
    probe_report["frozen_train_floors"] = {k: float(v) for k, v in floors.items()}
    probe_report["floor_pass"] = {k: observed[k] >= float(floors[k]) for k in floors}
    probe_report["all_floors_pass"] = all(probe_report["floor_pass"].values())
    decoder_report = report(decoded)
    token_fraction = lengths / lengths.sum(1, keepdim=True)
    decoder_report["importance_token_fraction_mae"] = float((importance-token_fraction).abs().mean())
    return {"exact4_documents": len(indexes), "built_in_parent_only_decoder": decoder_report,
            "train_fitted_parent_only_probe": probe_report, "probe_sha256": spec["probe_sha256"]}


def _bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    rng=np.random.default_rng(seed); n=len(values)
    estimates=[float(values[rng.integers(0,n,n)].mean()) for _ in range(replicates)]
    return [float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))]


def _distribution_geometry(target: np.ndarray, generated: np.ndarray) -> dict:
    residual = generated - target
    target_centered = target - target.mean(0); generated_centered = generated - generated.mean(0)
    target_cov = target_centered.T @ target_centered / max(1, len(target)-1)
    generated_cov = generated_centered.T @ generated_centered / max(1, len(generated)-1)
    target_std = target.std(0); generated_std = generated.std(0)
    singular = np.linalg.svd(residual, compute_uv=False)
    return {"mean_vector_l2": float(np.linalg.norm(generated.mean(0)-target.mean(0))),
            "mean_coordinate_bias": float(residual.mean()),
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
            "variance_ratio_mean": float(np.mean((generated_std**2)/(target_std**2+1e-12))),
            "std_ratio_mean": float(np.mean(generated_std/(target_std+1e-12))),
            "covariance_relative_frobenius": float(np.linalg.norm(generated_cov-target_cov)/(np.linalg.norm(target_cov)+1e-12)),
            "residual_effective_rank": float((singular.sum()**2)/(np.square(singular).sum()+1e-12)),
            "residual_top_singular_fraction": float(singular[0]/(singular.sum()+1e-12))}


def evaluate_safe_scopes(data: dict, cfg: dict, attack_cfg: dict) -> dict:
    scopes={"pooled":np.arange(len(data["roots"]))} | {c:np.flatnonzero(data["corpora"]==c) for c in CORPORA}
    result={}
    for offset,(scope,indexes) in enumerate(scopes.items()):
        if not len(indexes): raise ValueError(f"safe arm has no rows for {scope}")
        cosine=_cosine_values(data["roots"][indexes],data["targets"][indexes])
        result[scope]={"rows":int(len(indexes)),"teacher_cosine":float(cosine.mean()),
            "teacher_cosine_ci95_bootstrap":_bootstrap_mean(cosine,int(cfg["bootstrap_replicates"]),int(cfg["bootstrap_seed"])+1000+offset),
            "attacks":fixed_attacks(data["targets"][indexes],data["roots"][indexes],data["groups"][indexes],attack_cfg,int(cfg["bootstrap_replicates"]),int(cfg["bootstrap_seed"])+100*offset),
            "distribution_geometry":_distribution_geometry(data["targets"][indexes],data["roots"][indexes]),
            "health":_health(data["roots"][indexes],cfg)}
    return result


def evaluate_full_scopes(data: dict, cfg: dict) -> dict:
    scopes={"pooled":np.arange(len(data["roots"]))} | {c:np.flatnonzero(data["corpora"]==c) for c in CORPORA}
    return {scope:{"rows":int(len(indexes)),"health":_health(data["roots"][indexes],cfg)} for scope,indexes in scopes.items()}


def run(root: Path, protocol_path: Path, specification_path: Path, device_name: str="cuda") -> Path:
    """Evaluate once.  No cache preparation, corpus access, fitting, or training occurs."""
    protocol=yaml.safe_load(protocol_path.read_text()); validate_protocol(protocol)
    spec=yaml.safe_load(specification_path.read_text())
    if spec.get("v3_cls_experiment") is not True: raise ValueError("evaluation specification is not V3")
    output=root/spec["output_dir"]; _require_namespace(output,"evaluation output")
    result_path=output/"TWO_ARM_EVALUATION.json"; marker=output/"TWO_ARM_EVALUATION_STARTED"
    if result_path.exists() or marker.exists(): raise FileExistsError("V3 two-arm evaluation is create-only")
    output.mkdir(parents=True,exist_ok=True)
    fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,"w") as handle: handle.write(f"spec_sha256={sha256_file(specification_path)}\n")
    device=torch.device(device_name if device_name=="cpu" or torch.cuda.is_available() else "cpu")
    model,_,checkpoint_provenance=load_checkpoint(root,spec,device)
    safe_by,full_by,cache_provenance=load_two_arm_cache(root,spec,protocol)
    attack_path=root/protocol["attack_config"]; _require_hash(attack_path,protocol["attack_config_sha256"],"attack config")
    attack_cfg=yaml.safe_load(attack_path.read_text())
    safe=infer_safe(model,safe_by,device,int(protocol["batches"]["safe_root"]))
    full=infer_full(model,full_by,device,int(protocol["batches"]["full_merge"]))
    result={"study":"v3_cls","role":spec["role"],"seed":int(spec["seed"]),"training_performed":False,
        "safe_direct":{"definition":"1025..4096 exact-token prefix, direct normalized CLS target, one K4 merge","scopes":evaluate_safe_scopes(safe,protocol,attack_cfg),"telemetry":safe["telemetry"],"retention":evaluate_exact4_retention(model,safe,spec,root,device)},
        "full_document":{"definition":"every token, exact contiguous 1024-token leaves (final leaf may be short), recursive K4, no truncation","teacher_comparison_performed":False,"attack_evaluation_performed":False,"scopes":evaluate_full_scopes(full,protocol),"telemetry":full["telemetry"]},
        "provenance":{"checkpoint":checkpoint_provenance,"cache":cache_provenance,"protocol":str(protocol_path),"protocol_sha256":sha256_file(protocol_path),"specification":str(specification_path),"specification_sha256":sha256_file(specification_path),"attack_config":protocol["attack_config"],"attack_config_sha256":sha256_file(attack_path),"evaluator_sha256":sha256_file(__file__)}}
    temporary=result_path.with_suffix(".tmp"); temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); os.replace(temporary,result_path)
    (output/"TWO_ARM_EVALUATION_COMPLETE").write_text(f"{sha256_file(result_path)}  {result_path.name}\n")
    return result_path
