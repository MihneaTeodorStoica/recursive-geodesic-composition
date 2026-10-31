"""Integrity and I/O for the additive safe-baseline study, not an official rerun."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import uuid
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/safe_baselines_v1.yaml"
PUBLISHED_HASH = "2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return actual


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_protocol(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["study"] != "safe_baselines_v1" or cfg["merger_updates"] != 0:
        raise ValueError("wrong study or nonzero merger updates")
    if cfg["analysis_role"] != "additive_post_review_on_previously_evaluated_documents":
        raise ValueError("analysis must disclose reuse of the evaluation population")
    if cfg["methods"] != ["first_chunk", "unweighted_mean", "token_weighted_mean",
                           "learned_reference", "learned_parent"]:
        raise ValueError("representation set changed")
    if cfg["seeds"] != [424242, 271828, 314159] or cfg["bootstrap"]["reps"] != 2000:
        raise ValueError("seed/replicate protocol changed")
    for key in ("cache_dir", "output_dir"):
        if cfg[key] != {"cache_dir": "cache_safe_baselines_v1",
                        "output_dir": "results/safe_baselines_v1"}[key]:
            raise ValueError("refusing another cache/result namespace")
    if cfg["published_results_sha256"] != PUBLISHED_HASH:
        raise ValueError("published result identity changed")
    require_hash(ROOT / cfg["published_results"], PUBLISHED_HASH)
    require_hash(ROOT / cfg["attack_config"], cfg["attack_config_sha256"])
    official = yaml.safe_load((ROOT / cfg["official_config"]).read_text())
    if cfg["attack_config_sha256"] != official["attack_config_sha256"]:
        raise ValueError("classifier recipe differs from official recipe")
    published = json.loads((ROOT / cfg["published_results"]).read_text())
    cfg["config_sha256"] = sha256(path)
    return cfg, official, published


def environment(device):
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "datasets", "numpy", "scikit-learn", "scipy", "tokenizers")},
        "device": str(device), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": (torch.cuda.get_device_properties(device).total_memory
                             if device.type == "cuda" else None),
        "cpu_threads": torch.get_num_threads(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "git_commit": "anonymous-supplement-source",
    }


def configure_device(cfg, name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(int(cfg["inference"]["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["inference"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(cfg["inference"]["allow_tf32"])
    return device


def checkpoint_entries(cfg):
    manifest = json.loads((ROOT / cfg["release_manifest"]).read_text())
    entries = {entry["seed"]: entry for entry in manifest["artifacts"]
               if entry["role"] == "final_clean_v6_checkpoint"}
    result = []
    for seed in cfg["seeds"]:
        entry = dict(entries[seed])
        entry["path"] = str(Path(cfg["artifact_dir"]) / entry["asset_filename"])
        path = ROOT / entry["path"]
        if path.stat().st_size != entry["byte_size"]:
            raise ValueError(f"checkpoint size mismatch: {path}")
        require_hash(path, entry["sha256"])
        result.append(entry)
    return result


def source_hashes():
    files = [
        "configs/safe_baselines_v1.yaml", "configs/official_v6.yaml",
        "configs/evaluator_attacks_v1.yaml", "scripts/safe_baselines_protocol.py",
        "scripts/prepare_safe_baselines.py", "scripts/evaluate_safe_baselines.py",
        "src/recursive_geodesic_composition/safe_baselines.py",
        "src/recursive_geodesic_composition/architecture.py",
        "src/recursive_geodesic_composition/data.py",
        "src/recursive_geodesic_composition/evaluate_kary_validation.py",
        "scripts/official_v6/official_test_v6_protocol.py",
        "scripts/official_v6/evaluate_official_test_v6_recovery.py",
    ]
    return {name: sha256(ROOT / name) for name in files if (ROOT / name).exists()}


def validate_population(rows, corpus, published):
    expected = published["cache_provenance"]["corpora"][corpus]
    ids = [row["id"] for row in rows]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError(f"{corpus}: expected 1000 unique documents")
    if stable_hash(ids) != expected["document_order_sha256"]:
        raise ValueError(f"{corpus}: population differs from the published document order")
    for index, row in enumerate(rows):
        source_index = row["source_index"]
        if row["corpus"] != corpus or row["id"] != f"{corpus}:{source_index}":
            raise ValueError("invalid document identity")
        if not 162000 <= source_index < 192000 or row["selected_rank"] != index:
            raise ValueError("invalid source range/order")
        if not 1025 <= row["safe_prefix_token_count"] <= 4096:
            raise ValueError("invalid safe prefix length")
    if sum(row["safe_prefix_token_count"] for row in rows) != expected["safe_prefix_tokens"]["sum"]:
        raise ValueError(f"{corpus}: prefix token totals differ")


def load_safe_cache(cfg, published):
    from recursive_geodesic_composition.safe_baselines import validate_children, validate_vectors
    cache = ROOT / cfg["cache_dir"]
    manifest = json.loads((cache / "CACHE_MANIFEST.json").read_text())
    if (cache / "COMPLETE").read_text().strip() != sha256(cache / "CACHE_MANIFEST.json"):
        raise ValueError("cache completion hash mismatch")
    if manifest["config_sha256"] != cfg["config_sha256"]:
        raise ValueError("cache configuration changed")
    if manifest["published_results_sha256"] != PUBLISHED_HASH:
        raise ValueError("cache refers to another original result")
    rows = []
    for corpus in published["corpus_order"]:
        entry = manifest["corpora"][corpus]
        path = cache / corpus / "safe.pt"
        require_hash(path, entry["safe_sha256"])
        part = torch.load(path, map_location="cpu", weights_only=True)
        validate_population(part, corpus, published)
        for row in part:
            leaves, counts, target = row["leaves"], row["lengths"], row["teacher_target"]
            if leaves.shape != (len(counts), 768) or not 2 <= len(counts) <= 4 or target.shape != (768,):
                raise ValueError("invalid safe tensor shapes")
            if not torch.all(counts > 0) or not torch.all(counts <= 1024):
                raise ValueError("invalid token counts")
            if not torch.all(counts[:-1] == 1024) or counts.sum().item() != row["safe_prefix_token_count"]:
                raise ValueError("inexact chunk coverage")
            validate_children(leaves[None], counts[None])
            validate_vectors(leaves.numpy(), "leaves", unit=True)
            validate_vectors(target[None].numpy(), "teacher", unit=True)
        rows.extend(part)
    if stable_hash([row["id"] for row in rows]) != published["cache_provenance"]["pooled_document_order_sha256"]:
        raise ValueError("pooled document order changed")
    histogram = {k: sum(len(row["lengths"]) == k for row in rows) for k in (2, 3, 4)}
    if histogram != {2: 2707, 3: 687, 4: 606}:
        raise ValueError("active-arity population changed")
    return rows, manifest
