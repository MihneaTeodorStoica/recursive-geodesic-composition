"""Integrity and I/O for the additive safe-baseline study, not an official rerun."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import uuid
from pathlib import Path

import torch
import yaml

from public_replay import REPLAY_ID, package_file_hash, public_cache_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/safe_baselines_v1.yaml"
PUBLISHED_HASH = "2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917"
REPLAY_CONFIG = ROOT / "configs/safe_baselines_replay_v1.yaml"
REPLAY_CACHE = "cache_kary4_strict_autoencoder_official_test_v6"
REPLAY_OUTPUT = "replays/safe_baselines_v1"
HISTORICAL_METADATA = "results/safe_baselines_v1/DOCUMENT_METADATA.json"
HISTORICAL_CACHE_MANIFEST = "results/safe_baselines_v1/REGENERATED_CACHE_MANIFEST.json"


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
    if "execution_mode" in cfg:
        return load_replay_protocol(path, cfg)
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


def load_replay_protocol(path, replay):
    expected = {
        "execution_mode": REPLAY_ID,
        "study": "safe_baselines_v1",
        "scientific_config": "configs/safe_baselines_v1.yaml",
        "scientific_config_sha256": "5918162673076915f97aaf639acfac96465cdb8f239d244add5db93eaff30521",
        "source_cache": REPLAY_CACHE,
        "output_dir": REPLAY_OUTPUT,
    }
    if replay != expected or Path(path).resolve() != REPLAY_CONFIG.resolve():
        raise ValueError("invalid cached-replay baseline configuration or namespace")
    for name in (expected["scientific_config"], "configs/official_v6.yaml",
                 "configs/evaluator_attacks_v1.yaml", "reproducibility/ARTIFACT_MANIFEST.json"):
        digest = package_file_hash(ROOT, name)
        if name == expected["scientific_config"] and digest != expected["scientific_config_sha256"]:
            raise ValueError("historical baseline scientific configuration changed")
    cfg, official, published = load_protocol(DEFAULT_CONFIG)
    cfg.update({
        "execution_mode": REPLAY_ID,
        "replay_config": str(REPLAY_CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(path),
        "scientific_config": expected["scientific_config"],
        "scientific_config_sha256": expected["scientific_config_sha256"],
        "cache_dir": REPLAY_CACHE,
        "input_kind": "read_only_public_official_safe_tensor_view",
        "output_dir": REPLAY_OUTPUT,
    })
    return cfg, official, published


def environment(device):
    packages = {}
    optional_teacher_packages = {"transformers", "datasets", "tokenizers"}
    for name in ("torch", "transformers", "datasets", "numpy", "scikit-learn", "scipy", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name not in optional_teacher_packages:
                raise
            packages[name] = "not installed"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages,
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


def replay_source_hashes():
    names = ("configs/safe_baselines_replay_v1.yaml", "scripts/public_replay.py")
    return source_hashes() | {name: sha256(ROOT / name) for name in names}


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
    if cfg.get("execution_mode") == REPLAY_ID:
        return load_replay_safe_cache(cfg, published)
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


def load_replay_safe_cache(cfg, published):
    """Validate a read-only view of the released official FP32 safe tensors."""
    from recursive_geodesic_composition.recursive_reuse import k4_accounting
    from recursive_geodesic_composition.safe_baselines import validate_children, validate_vectors

    if cfg["cache_dir"] != REPLAY_CACHE or cfg["output_dir"] != REPLAY_OUTPUT:
        raise ValueError("cached-replay input/output namespace changed")
    cache = ROOT / REPLAY_CACHE
    manifest, release = public_cache_manifest(ROOT, REPLAY_CACHE)
    expected_manifest = {
        "stage": "kary4_strict_autoencoder_official_test_v6",
        "cache_namespace": "kary4_strict_autoencoder_official_test_v6",
        "protocol_version": 6,
        "roles": ["test"],
        "official_test": True,
        "complete": True,
        "declared_scan_range": [162000, 192000],
        "selection": "first_1000_safe_prefix_length_eligible_only",
        "eligible_documents_per_corpus": 1000,
        "same_accepted_documents_both_arms": True,
        "arity": 4,
        "chunk_tokens": 1024,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"public official cache manifest mismatch: {key}")
    corpus_order = published["corpus_order"]
    if set(manifest["corpora"]) != set(corpus_order):
        raise ValueError("public cache corpus inventory changed")

    metadata_hash = package_file_hash(ROOT, HISTORICAL_METADATA)
    historical_metadata = json.loads((ROOT / HISTORICAL_METADATA).read_text())
    historical_cache_hash = package_file_hash(ROOT, HISTORICAL_CACHE_MANIFEST)
    historical_cache = json.loads((ROOT / HISTORICAL_CACHE_MANIFEST).read_text())
    if len(historical_metadata) != 4000 or historical_cache["config_sha256"] != cfg["scientific_config_sha256"]:
        raise ValueError("historical baseline metadata/cache identity changed")

    rows, files, source_identities = [], {}, set()
    for corpus in corpus_order:
        entry = manifest["corpora"][corpus]
        original = published["cache_provenance"]["corpora"][corpus]
        safe_name, index_name = f"{corpus}/safe.pt", f"{corpus}/full_index.json"
        for name, field in ((safe_name, "safe_tensor"), (index_name, "full_index")):
            digest = entry[f"{field}_sha256"]
            if (entry[field] != name or manifest["artifacts"].get(name) != digest or
                    original[field] != f"{REPLAY_CACHE}/{name}" or
                    original[f"{field}_sha256"] != digest):
                raise ValueError(f"{corpus}: published {field} identity mismatch")
            require_hash(cache / name, digest)
        part = torch.load(cache / safe_name, map_location="cpu", weights_only=True)
        index = json.loads((cache / index_name).read_text())
        if (not isinstance(part, list) or
                (index.get("cache_namespace"), index.get("cache_role"), index.get("corpus")) !=
                (expected_manifest["cache_namespace"], "test", corpus)):
            raise ValueError(f"{corpus}: invalid official safe/full index schema")
        indexed = index["documents"]
        if len(part) != 1000 or len(indexed) != 1000:
            raise ValueError(f"{corpus}: expected 1000 matching safe and full records")
        validate_population(part, corpus, published)
        indexes = []
        for rank, (row, full) in enumerate(zip(part, indexed)):
            for key, value in {
                "cache_namespace": expected_manifest["cache_namespace"],
                "cache_role": "test", "arm": "safe_teacher_comparable",
                "reduction": "direct_single_k4", "arity": 4, "chunk_tokens": 1024,
                "teacher_max_tokens": 4096,
            }.items():
                if row.get(key) != value:
                    raise ValueError(f"{row['id']}: official safe field changed: {key}")
            for key in ("id", "source_index", "selected_rank", "source_document_id", "corpus",
                        "full_token_count", "safe_prefix_token_count", "full_token_ids_sha256"):
                if row.get(key) != full.get(key):
                    raise ValueError(f"{corpus}/{rank}: safe/full identity differs: {key}")
            source_id = row["source_document_id"]
            if not isinstance(source_id, str) or not source_id or (corpus, source_id) in source_identities:
                raise ValueError(f"{corpus}/{rank}: missing or duplicate source identity")
            source_identities.add((corpus, source_id))
            indexes.append(row["source_index"])
            prefix, total = row["safe_prefix_token_count"], row["full_token_count"]
            if (isinstance(prefix, bool) or not isinstance(prefix, int) or
                    isinstance(total, bool) or not isinstance(total, int) or
                    prefix != min(total, 4096)):
                raise ValueError(f"{row['id']}: safe prefix is not the exact full-document prefix")
            for key in ("safe_prefix_token_ids_sha256", "full_token_ids_sha256"):
                if not isinstance(row.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None:
                    raise ValueError(f"{row['id']}: missing or malformed token identity: {key}")
            if prefix == total and row["safe_prefix_token_ids_sha256"] != row["full_token_ids_sha256"]:
                raise ValueError(f"{row['id']}: complete-prefix token identities differ")
            leaves, counts, target = row["leaves"], row["lengths"], row["teacher_target"]
            if (not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                        for value in (leaves, counts, target)) or
                    counts.ndim != 1 or leaves.shape != (len(counts), 768) or target.shape != (768,) or
                    not 2 <= len(counts) <= 4):
                raise ValueError(f"{row['id']}: invalid FP32 safe tensors")
            if (not torch.isfinite(counts).all() or not torch.equal(counts, counts.round()) or
                    not torch.all((counts > 0) & (counts <= 1024)) or
                    not torch.all(counts[:-1] == 1024) or counts.sum().item() != prefix):
                raise ValueError(f"{row['id']}: invalid exact chunk coverage")
            validate_children(leaves[None], counts[None])
            validate_vectors(leaves.numpy(), "official leaves", unit=True)
            validate_vectors(target[None].numpy(), "official teacher target", unit=True)
            full_leaves, depth, _ = k4_accounting(total)
            if (full["safe_leaf_count"] != len(counts) or full["full_leaf_count"] != full_leaves or
                    full["tree_depth"] != depth):
                raise ValueError(f"{row['id']}: full/safe leaf or depth accounting changed")
            metadata = {key: row[key] for key in (
                "id", "corpus", "source_index", "safe_prefix_token_count", "full_token_count",
                "safe_prefix_token_ids_sha256")} | {"active_children": len(counts)}
            if metadata != historical_metadata[len(rows) + rank]:
                raise ValueError(f"{row['id']}: archived baseline population/token identity differs")
        actual_range = entry["actual_scanned_range"]
        if (indexes != sorted(set(indexes)) or actual_range[0] != 162000 or
                not 162000 < actual_range[1] <= 192000 or indexes[-1] != actual_range[1] - 1 or
                stable_hash(indexes) != entry["source_index_order_sha256"] or
                stable_hash([row["id"] for row in part]) != entry["safe_and_full_document_order_sha256"]):
            raise ValueError(f"{corpus}: source scan/order identity changed")
        files[corpus] = {
            "official_safe_path": f"{REPLAY_CACHE}/{safe_name}",
            "official_safe_sha256": entry["safe_tensor_sha256"],
            "official_full_index_sha256": entry["full_index_sha256"],
            "archived_regenerated_safe_sha256": historical_cache["corpora"][corpus]["safe_sha256"],
            "official_and_regenerated_sha256_match": (
                entry["safe_tensor_sha256"] == historical_cache["corpora"][corpus]["safe_sha256"]),
        }
        rows.extend(part)

    if (len(rows) != 4000 or stable_hash([row["id"] for row in rows]) !=
            published["cache_provenance"]["pooled_document_order_sha256"]):
        raise ValueError("pooled official population/order changed")
    arities = Counter(len(row["lengths"]) for row in rows)
    if arities != {2: 2707, 3: 687, 4: 606}:
        raise ValueError("active-arity histogram changed")
    totals = [row["full_token_count"] for row in rows]
    accounting = [k4_accounting(total) for total in totals]
    histograms = {
        "token_count": Counter(totals),
        "leaf_count": Counter(item[0] for item in accounting),
        "tree_depth": Counter(item[1] for item in accounting),
    }
    for seed in published["seeds"]:
        original = seed["full_document_arm"]["document_distributions"]
        for name, values in histograms.items():
            if {str(key): count for key, count in sorted(values.items())} != original[name]["histogram"]:
                raise ValueError(f"full-document {name} histogram changed")

    view = {
        "execution_mode": REPLAY_ID,
        "view_kind": "read_only_public_official_safe_tensors_not_regenerated_baseline_cache",
        "source_release": release,
        "source_official_manifest_sha256": release["manifest_sha256"],
        "historical_official_manifest_sha256": published["cache_provenance"]["manifest_sha256"],
        "official_safe_files": files,
        "historical_regenerated_cache_manifest": HISTORICAL_CACHE_MANIFEST,
        "historical_regenerated_cache_manifest_sha256": historical_cache_hash,
        "historical_document_metadata": HISTORICAL_METADATA,
        "historical_document_metadata_sha256": metadata_hash,
        "documents": len(rows),
        "documents_per_corpus": 1000,
        "active_children_histogram": {str(key): arities[key] for key in (2, 3, 4)},
        "pooled_document_order_sha256": stable_hash([row["id"] for row in rows]),
        "all_prefix_token_hashes_match_historical_metadata": True,
        "all_full_token_hashes_match_official_index": True,
        "full_population_histograms_match_all_official_seeds": True,
        "tensor_values": "unchanged FP32 leaves, lengths and teacher targets from public official safe.pt files",
        "batching_distinction": "official leaves came from full-document embedding batches of up to 16; the unavailable historical baseline cache used safe-only batches of up to four",
        "historical_regenerated_cache_recovered": False,
        "teacher_encoding_calls": 0,
        "corpus_downloads": 0,
    }
    return rows, view
