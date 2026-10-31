"""Historical official-v6 recovery and separately identified cached replay."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from recursive_geodesic_composition.evaluate_kary_validation import ParentOnlyRetentionProbe, build_model
from recursive_geodesic_composition.validation_output_calibration import diagnostics, orientation_safe_auc
from prepare_official_test_v6 import validate_full_shard, validate_safe_row
from official_test_v6_protocol import (
    ARITY,
    CACHE_NAMESPACE,
    CHUNK_TOKENS,
    CONFIG,
    CORPUS_ORDER,
    PRIOR_RANGES,
    ROLE,
    ROOT,
    SCAN,
    SEED_ORDER,
    TARGET_DOCUMENTS,
    atomic_text,
    claim_once,
    fsync_directory,
    load_config,
    reduction_depth,
    require_hash,
    require_storage,
    sha256_file,
    stable_hash,
    validate_immutable_provenance,
)

ATTACK_NAMES = ("standard_scaled_logistic", "standard_scaled_rbf_svm")
SCOPES = ("pooled", *CORPUS_ORDER)

# The anonymous replay is a separate release contract. These are identities of
# the public inputs and retained historical evidence, not replacement hashes
# for the pre-access files.
REPLAY_CONFIG_PATH = "configs/official_v6.yaml"
REPLAY_CONFIG_SHA256 = "7d32da2950040dcd7f86a863aac5e941db3628cd15aae83288bdb3756c6a1baa"
REPLAY_INVENTORY_SHA256 = "4b9ab6398cc3dcc1d8ae735ee01b7bcc3f33586a5cf928f37dc5306dd585a435"
REPLAY_CACHE_SHA256 = "833d3cf216ac6349be0b21b070e92e0f39063d59623d499e5cd67f54d108d0ed"
REPLAY_ARCHIVE_SHA256 = "b615e66e3f9595d01e4fae0af0dfcb17dfe6a78490c8756982e6886ba7d3dc30"
REPLAY_CACHE_REVISION = "9e4c8e02375f66b3cf9918a20eb81446f996620e"
HISTORICAL_RESULT_PATH = "results/official_v6/results.json"
HISTORICAL_RESULT_SHA256 = "2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917"
PREDECLARATION_PATH = "reproducibility/protocols/official_v6/PREDECLARATION_SHA256SUMS"
PREDECLARATION_SHA256 = "0b74753c2a42113e924d9dbb7630297dd4e1361440c3906707fb732fa23c068b"
RECOVERY_PATH = "reproducibility/protocols/official_v6/RECOVERY_SHA256SUMS"
RECOVERY_SHA256 = "5a80aa7493e603da544ecba3942f33dc54ec4a31952cb0f0d8a5baa9a6b0766b"


def require_finite(value, label: str) -> None:
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError(f"nonfinite value in {label}")


def _summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    require_finite(values, "distribution")
    return {
        "count": int(values.size),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(values.max()),
    }


def _histogram(values) -> dict:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique, counts)}


class RunningStats:
    """Constant-memory exact count/moments/extrema for unrestricted recursion."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(array):
            return
        require_finite(array, "streaming metric")
        self.count += len(array)
        self.total += float(array.sum())
        self.total_square += float(np.square(array).sum())
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def result(self) -> dict:
        if self.count <= 0:
            raise ValueError("missing required streaming metric observations")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def _health(values: np.ndarray, collapse_cfg: dict, label: str) -> dict:
    values = np.asarray(values, dtype=np.float32)
    require_finite(values, f"{label} roots")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    require_finite(norms, f"{label} norms")
    health = diagnostics(values, norms, {"collapse": collapse_cfg})
    health["output_norm_saturation_count"] = int(health.pop("adapter_saturation_count"))
    centered = values.astype(np.float64) - values.astype(np.float64).mean(axis=0)
    health["centered_numeric_rank"] = int(np.linalg.matrix_rank(centered))
    health["all_finite"] = bool(health["finite_fraction"] == 1.0)
    return health


def validate_cache_layout(cache: Path) -> None:
    expected_root = {"CACHE_MANIFEST.json", "COMPLETE", *CORPUS_ORDER}
    if not cache.is_dir() or {path.name for path in cache.iterdir()} != expected_root:
        raise ValueError("v6 cache root has missing or undeclared role/artifacts")
    for corpus in CORPUS_ORDER:
        directory = cache / corpus
        if {path.name for path in directory.iterdir()} != {
            "safe.pt",
            "full_index.json",
            "full",
        }:
            raise ValueError(f"{corpus} cache layout is not direct safe/full test-only")
        shards = sorted((directory / "full").iterdir())
        if len(shards) != TARGET_DOCUMENTS or any(
            path.name != f"{index:06d}.pt" for index, path in enumerate(shards)
        ):
            raise ValueError(
                f"{corpus} full shards are missing, extra, or out of order"
            )


def _pinned_package_input(relative: str, expected: str) -> str:
    from public_replay import package_file_hash

    actual = package_file_hash(ROOT, relative)
    if actual != expected:
        raise ValueError(f"anonymous official replay input identity changed: {relative}")
    return actual


def _checksum_ledger(relative: str, expected: str) -> dict[str, str]:
    _pinned_package_input(relative, expected)
    entries = {}
    for line in (ROOT / relative).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, path = line.split(maxsplit=1)
        if (
            path in entries
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"malformed historical checksum ledger: {relative}")
        entries[path] = digest
    return entries


def _require_ledger_identity(ledger: dict, path: str, digest: str) -> None:
    if ledger.get(path) != digest:
        raise ValueError(f"historical identity is not recorded in its ledger: {path}")


def _historical_result() -> dict:
    _pinned_package_input(HISTORICAL_RESULT_PATH, HISTORICAL_RESULT_SHA256)
    return json.loads((ROOT / HISTORICAL_RESULT_PATH).read_text())


def _validate_public_cache_contract(
    cfg: dict, manifest: dict, release: dict, normalized: dict
) -> tuple[dict, dict]:
    """Bind the derivative cache to both the public inputs and old evidence."""
    from public_replay import REPLAY_ID, development_freeze, package_file_hash

    config_hash = _pinned_package_input(REPLAY_CONFIG_PATH, REPLAY_CONFIG_SHA256)
    if cfg != load_config(CONFIG):
        raise ValueError("replay configuration differs from the pinned public config")
    if (
        release.get("replay_contract") != REPLAY_ID
        or release.get("repository")
        != "tmlr-embedding-artifacts/recursive-geodesic-checkpoints"
        or release.get("revision") != REPLAY_CACHE_REVISION
        or release.get("inventory_sha256") != REPLAY_INVENTORY_SHA256
        or release.get("archive_path") != f"caches/{cfg['cache_dir']}.tar.gz"
        or release.get("archive_sha256") != REPLAY_ARCHIVE_SHA256
        or release.get("manifest_path") != f"{cfg['cache_dir']}/CACHE_MANIFEST.json"
        or release.get("manifest_sha256") != REPLAY_CACHE_SHA256
        or release.get("verified_files") != 4010
        or release.get("historical_identities_rebound") is not False
    ):
        raise ValueError("official cache is not the pinned anonymous replay release")

    predeclared = _checksum_ledger(PREDECLARATION_PATH, PREDECLARATION_SHA256)
    recovery = _checksum_ledger(RECOVERY_PATH, RECOVERY_SHA256)
    historical = _historical_result()
    expected_history = {
        "stage": CACHE_NAMESPACE,
        "protocol_version": 6,
        "seed_order": list(SEED_ORDER),
        "corpus_order": list(CORPUS_ORDER),
        "declared_scan_window_per_corpus": list(SCAN),
        "same_accepted_documents_both_arms": True,
        "attack_config_sha256": cfg["attack_config_sha256"],
        "bootstrap": json.loads(json.dumps(cfg["bootstrap"])),
    }
    for key, expected in expected_history.items():
        if historical.get(key) != expected:
            raise ValueError(f"historical official scientific contract changed: {key}")

    old = manifest.get("immutable_provenance")
    if not isinstance(old, dict) or old != historical.get("immutable_provenance"):
        raise ValueError("cache and historical result provenance differ")
    if set(old) != set(normalized):
        raise ValueError("historical and normalized provenance fields differ")
    differing_metadata = {
        "checkpoints",
        "development_final_results",
        "development_final_results_sha256",
        "post_final_verification",
    }
    for key in set(old) - differing_metadata:
        if old[key] != normalized[key]:
            raise ValueError(f"historical and normalized scientific provenance differ: {key}")

    historical_config = {
        "path": manifest.get("config_path"),
        "sha256": manifest.get("config_sha256"),
    }
    for ledger in (predeclared, recovery):
        _require_ledger_identity(
            ledger, historical_config["path"], historical_config["sha256"]
        )
    _require_ledger_identity(
        predeclared, cfg["attack_config"], cfg["attack_config_sha256"]
    )
    metadata = {
        "historical_config": historical_config,
        "normalized_config": {"path": REPLAY_CONFIG_PATH, "sha256": config_hash},
        "historical_result": {
            "path": HISTORICAL_RESULT_PATH,
            "sha256": HISTORICAL_RESULT_SHA256,
        },
        "predeclaration_ledger": {
            "path": PREDECLARATION_PATH,
            "sha256": PREDECLARATION_SHA256,
        },
        "recovery_ledger": {"path": RECOVERY_PATH, "sha256": RECOVERY_SHA256},
        "development_records": {},
        "seed_freezes": {},
        "historical_identities_are_evidence_not_rebound_inputs": True,
    }
    for name in ("development_final_results", "post_final_verification"):
        public_name = "final_results" if name == "development_final_results" else name
        old_path, old_hash = old[name], old[f"{name}_sha256"]
        for ledger in (predeclared, recovery):
            _require_ledger_identity(ledger, old_path, old_hash)
        public_path = cfg["development_provenance"][public_name]
        public_hash = cfg["development_provenance"][f"{public_name}_sha256"]
        if (
            package_file_hash(ROOT, public_path) != public_hash
            or normalized[name] != public_path
            or normalized[f"{name}_sha256"] != public_hash
        ):
            raise ValueError(f"normalized development identity changed: {name}")
        metadata["development_records"][name] = {
            "historical": {"path": old_path, "sha256": old_hash},
            "normalized": {"path": public_path, "sha256": public_hash},
        }

    if set(old["checkpoints"]) != {str(seed) for seed in SEED_ORDER}:
        raise ValueError("historical checkpoint seed set changed")
    if set(normalized["checkpoints"]) != set(SEED_ORDER):
        raise ValueError("normalized checkpoint seed set changed")
    for seed in SEED_ORDER:
        previous = old["checkpoints"][str(seed)]
        current = normalized["checkpoints"][seed]
        if set(previous) != set(current):
            raise ValueError(f"seed {seed} checkpoint provenance fields differ")
        for key in set(previous) - {"freeze", "freeze_sha256"}:
            if previous[key] != current[key]:
                raise ValueError(f"seed {seed} scientific provenance differs: {key}")
        _require_ledger_identity(
            predeclared, previous["freeze"], previous["freeze_sha256"]
        )
        for name in ("checkpoint", "source_checkpoint"):
            _require_ledger_identity(
                predeclared, previous[name], previous[f"{name}_sha256"]
            )
        for probe in previous["train_probes"].values():
            _require_ledger_identity(predeclared, probe["path"], probe["sha256"])
        _, freeze_identity = development_freeze(ROOT, seed)
        if (
            freeze_identity["normalized_freeze_path"] != current["freeze"]
            or freeze_identity["normalized_freeze_sha256"] != current["freeze_sha256"]
            or freeze_identity["public_protocol_sha256"] != config_hash
        ):
            raise ValueError(f"seed {seed} normalized freeze identity changed")
        metadata["seed_freezes"][str(seed)] = {
            "historical": {
                "path": previous["freeze"],
                "sha256": previous["freeze_sha256"],
            },
            "normalized": freeze_identity,
        }

    inventory = json.loads((ROOT / release["inventory_path"]).read_text())
    bundles = [
        item
        for item in inventory["inventory"]["archives"]
        if item["cache_directory"] == cfg["cache_dir"]
    ]
    if len(bundles) != 1:
        raise ValueError("official cache release inventory is ambiguous")
    cache_identities = {}
    for filename in ("CACHE_MANIFEST.json", "COMPLETE"):
        path = f"{cfg['cache_dir']}/{filename}"
        entries = [item for item in bundles[0]["files"] if item["path"] == path]
        if len(entries) != 1:
            raise ValueError(f"missing cache metadata derivative identity: {filename}")
        entry = entries[0]
        cache_identities[filename] = {
            "historical": {
                "path": path,
                "bytes": entry["original_bytes"],
                "sha256": entry["original_sha256"],
            },
            "published": {
                "path": path,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            },
            "transformation": entry["transformation"],
        }
    old_manifest_hash = cache_identities["CACHE_MANIFEST.json"]["historical"]["sha256"]
    if (
        cache_identities["CACHE_MANIFEST.json"]["published"]["sha256"]
        != release["manifest_sha256"]
        or historical["cache_provenance"]["manifest_sha256"] != old_manifest_hash
    ):
        raise ValueError("historical and published cache manifest identities disagree")
    _require_ledger_identity(
        recovery, f"{cfg['cache_dir']}/CACHE_MANIFEST.json", old_manifest_hash
    )
    metadata["cache_metadata_derivatives"] = cache_identities
    return {"release": release, "metadata_identities": metadata}, historical


def load_cache(
    cfg: dict, *, replay_immutable: dict | None = None
) -> tuple[dict[str, list], dict[str, list], dict]:
    cache = ROOT / cfg["cache_dir"]
    validate_cache_layout(cache)
    manifest_path, complete_path = cache / "CACHE_MANIFEST.json", cache / "COMPLETE"
    manifest_hash = sha256_file(manifest_path)
    if complete_path.read_text().strip() != f"{manifest_hash}  CACHE_MANIFEST.json":
        raise ValueError("atomic cache COMPLETE marker does not match manifest")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "stage": CACHE_NAMESPACE,
        "protocol_version": 6,
        "cache_namespace": CACHE_NAMESPACE,
        "roles": [ROLE],
        "official_test": True,
        "complete": True,
        "declared_scan_range": list(SCAN),
        "prior_ranges_refused": {
            key: list(value) for key, value in PRIOR_RANGES.items()
        },
        "selection": "first_1000_safe_prefix_length_eligible_only",
        "eligible_documents_per_corpus": TARGET_DOCUMENTS,
        "same_accepted_documents_both_arms": True,
        "arity": ARITY,
        "chunk_tokens": CHUNK_TOKENS,
        "safe_arm": cfg["safe_arm"],
        "full_arm": cfg["full_arm"],
    }
    if replay_immutable is None:
        expected["config_path"] = str(CONFIG.relative_to(ROOT))
        expected["config_sha256"] = sha256_file(CONFIG)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"v6 cache manifest mismatch: {key}")
    replay_contract, historical = None, None
    if replay_immutable is None:
        immutable = manifest.get("immutable_provenance", {})
        development = cfg["development_provenance"]
        if (
            immutable.get("development_final_results_sha256")
            != development["final_results_sha256"]
            or immutable.get("post_final_verification_sha256")
            != development["post_final_verification_sha256"]
            or immutable.get("implementation_parent_commit")
            != cfg["implementation_parent_commit"]
            or immutable.get("protected_data_accessed_before_freeze") is not False
        ):
            raise ValueError(
                "cache was not prepared under the clean-v6 development provenance"
            )
    else:
        from public_replay import public_cache_manifest

        published_manifest, release = public_cache_manifest(ROOT, cfg["cache_dir"])
        if published_manifest != manifest or release["manifest_sha256"] != manifest_hash:
            raise ValueError("public cache manifest changed during replay preflight")
        replay_contract, historical = _validate_public_cache_contract(
            cfg, manifest, release, replay_immutable
        )

    safe_by, full_by, all_groups = {}, {}, set()
    provenance = {"manifest_sha256": manifest_hash, "corpora": {}}
    for corpus in CORPUS_ORDER:
        summary = manifest.get("corpora", {}).get(corpus, {})
        safe_path = cache / summary.get("safe_tensor", "")
        index_path = cache / summary.get("full_index", "")
        safe_hash = require_hash(
            safe_path, summary.get("safe_tensor_sha256", ""), f"{corpus} safe tensor"
        )
        index_hash = require_hash(
            index_path, summary.get("full_index_sha256", ""), f"{corpus} full index"
        )
        if (
            manifest.get("artifacts", {}).get(str(safe_path.relative_to(cache)))
            != safe_hash
        ):
            raise ValueError(f"{corpus} safe manifest hash mismatch")
        if (
            manifest.get("artifacts", {}).get(str(index_path.relative_to(cache)))
            != index_hash
        ):
            raise ValueError(f"{corpus} full-index manifest hash mismatch")
        safe_rows = torch.load(safe_path, map_location="cpu", weights_only=False)
        index = json.loads(index_path.read_text())
        full_rows = index.get("documents", ())
        if (
            index.get("cache_namespace"),
            index.get("cache_role"),
            index.get("corpus"),
        ) != (CACHE_NAMESPACE, ROLE, corpus):
            raise ValueError(f"{corpus} full index namespace/role mismatch")
        if (
            not isinstance(safe_rows, list)
            or len(safe_rows) != TARGET_DOCUMENTS
            or len(full_rows) != TARGET_DOCUMENTS
        ):
            raise ValueError(
                f"{corpus} must contain exactly 1000 documents in both arms"
            )
        actual_end = int(summary.get("actual_scanned_range", [-1, -1])[1])
        if (
            summary.get("actual_scanned_range", [None])[0] != SCAN[0]
            or not SCAN[0] < actual_end <= SCAN[1]
        ):
            raise ValueError(f"{corpus} actual scan escaped v6 range")
        groups, indexes = [], []
        for rank, (safe, full) in enumerate(zip(safe_rows, full_rows)):
            validate_safe_row(safe, corpus, actual_end)
            required = {
                "id": safe["id"],
                "source_index": safe["source_index"],
                "source_document_id": safe["source_document_id"],
                "corpus": corpus,
                "full_token_count": safe["full_token_count"],
                "full_leaf_count": int(full["full_leaf_count"]),
                "tree_depth": int(full["tree_depth"]),
            }
            if (
                int(safe.get("selected_rank", -1)) != rank
                or int(full.get("selected_rank", -1)) != rank
            ):
                raise ValueError("safe/full selected rank mismatch")
            for key in (
                "id",
                "source_index",
                "source_document_id",
                "corpus",
                "full_token_count",
            ):
                if full.get(key) != required[key]:
                    raise ValueError(
                        f"safe/full accepted document mismatch: {corpus}/{rank}/{key}"
                    )
            if safe.get("full_token_ids_sha256") != full.get("full_token_ids_sha256"):
                raise ValueError("safe/full token identity hash mismatch")
            if (
                int(full["full_leaf_count"])
                != (int(full["full_token_count"]) + CHUNK_TOKENS - 1) // CHUNK_TOKENS
            ):
                raise ValueError("full index leaf count is not exact")
            if int(full["tree_depth"]) != reduction_depth(int(full["full_leaf_count"])):
                raise ValueError("full index depth is not exact")
            shard = cache / str(full.get("shard", ""))
            if shard.resolve().parent != (cache / corpus / "full").resolve():
                raise ValueError("full shard escaped its direct corpus namespace")
            if shard.stat().st_size != int(full.get("shard_bytes", -1)):
                raise ValueError("full shard byte count mismatch")
            require_hash(
                shard, full.get("shard_sha256", ""), f"{corpus} full shard {rank}"
            )
            if replay_immutable is not None:
                # The inventory has authenticated the bytes; validate every
                # unrestricted shard before creating the replay output.
                shard_data = torch.load(shard, map_location="cpu", weights_only=False)
                validate_full_shard(shard_data, required)
                del shard_data
            group = str(safe["id"])
            if group in all_groups:
                raise ValueError(f"duplicate pooled source group: {group}")
            all_groups.add(group)
            groups.append(group)
            indexes.append(int(safe["source_index"]))
        if (
            indexes != sorted(indexes)
            or len(indexes) != len(set(indexes))
            or indexes[-1] != actual_end - 1
        ):
            raise ValueError(f"{corpus} is not deterministic first-1000 source order")
        if stable_hash(groups) != summary.get("safe_and_full_document_order_sha256"):
            raise ValueError(f"{corpus} safe/full document-order hash mismatch")
        if stable_hash(indexes) != summary.get("source_index_order_sha256"):
            raise ValueError(f"{corpus} source-index order hash mismatch")
        safe_by[corpus], full_by[corpus] = safe_rows, list(full_rows)
        provenance["corpora"][corpus] = {
            "safe_tensor": str(safe_path.relative_to(ROOT)),
            "safe_tensor_sha256": safe_hash,
            "full_index": str(index_path.relative_to(ROOT)),
            "full_index_sha256": index_hash,
            "full_shards_verified": len(full_rows),
            "document_order_sha256": stable_hash(groups),
            "actual_scanned_range": [SCAN[0], actual_end],
            "selection_counts": {
                key: summary[key]
                for key in (
                    "scanned",
                    "accepted",
                    "excluded_blank_or_nontext",
                    "excluded_safe_prefix_too_short",
                )
            },
            "preparation_throughput": summary["preparation_throughput"],
            "full_tokens": summary["full_tokens"],
            "full_leaves": summary["full_leaves"],
            "full_depth": summary["full_depth"],
            "safe_prefix_tokens": summary["safe_prefix_tokens"],
        }
    provenance["pooled_document_order_sha256"] = stable_hash(
        [row["id"] for corpus in CORPUS_ORDER for row in safe_by[corpus]]
    )
    if historical is not None:
        old_population = historical["cache_provenance"]
        if (
            provenance["corpora"] != old_population["corpora"]
            or provenance["pooled_document_order_sha256"]
            != old_population["pooled_document_order_sha256"]
        ):
            raise ValueError("public cache population differs from historical evidence")
        provenance["public_replay"] = replay_contract
        provenance["preparation_throughput_is_historical"] = True
    return safe_by, full_by, provenance


def grouped_attack_split(real, fake, groups, attack_cfg: dict):
    real, fake, groups = np.asarray(real), np.asarray(fake), np.asarray(groups)
    if real.shape != fake.shape or real.ndim != 2 or len(groups) != len(real):
        raise ValueError("fixed attacks require matched rank-2 teacher/model arrays")
    if len(set(map(str, groups))) != len(groups):
        raise ValueError("fixed attacks require unique source-document groups")
    require_finite(real, "teacher attack embeddings")
    require_finite(fake, "AE attack embeddings")
    x = np.vstack((real, fake))
    y = np.r_[np.ones(len(real), dtype=np.int8), np.zeros(len(fake), dtype=np.int8)]
    paired = np.r_[groups, groups]
    train, holdout = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=float(attack_cfg["test_size"]),
            random_state=int(attack_cfg["split_seed"]),
        ).split(x, y, paired)
    )
    if set(map(str, paired[train])) & set(map(str, paired[holdout])):
        raise AssertionError("source group leaked across attack split")
    if set(np.unique(y[train])) != {0, 1} or set(np.unique(y[holdout])) != {0, 1}:
        raise ValueError("attack split lost a class")
    return x, y, paired, train, holdout


def document_group_bootstrap(y, scores, groups, reps: int, seed: int) -> np.ndarray:
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    unique = np.unique(groups)
    if int(reps) != 2000 or len(unique) < 2:
        raise ValueError("official bootstrap requires exactly 2000 reps and >=2 groups")
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng, values = np.random.default_rng(int(seed)), []
    for _ in range(int(reps)):
        sampled = rng.choice(unique, len(unique), replace=True)
        indexes = np.concatenate([members[group] for group in sampled])
        if set(np.unique(y[indexes])) != {0, 1}:
            raise AssertionError("paired source bootstrap lost a class")
        values.append(
            orientation_safe_auc(float(roc_auc_score(y[indexes], scores[indexes])))
        )
    result = np.asarray(values, dtype=np.float64)
    require_finite(result, "attack bootstrap")
    return result


def _ci(values) -> list[float]:
    return [float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5])]


def evaluate_fixed_attacks(real, fake, groups, attack_cfg: dict, reps: int, seed: int):
    x, y, paired, train, holdout = grouped_attack_split(real, fake, groups, attack_cfg)
    model_seed = int(attack_cfg["model_seed"])
    models = {
        "standard_scaled_logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(attack_cfg["logistic_c"]),
                        max_iter=int(attack_cfg["logistic_max_iter"]),
                        solver="lbfgs",
                        random_state=model_seed,
                    ),
                ),
            ]
        ),
        "standard_scaled_rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=float(attack_cfg["rbf_c"]),
                        gamma=float(attack_cfg["rbf_gamma"]),
                        kernel="rbf",
                        random_state=model_seed,
                    ),
                ),
            ]
        ),
    }
    metrics, bundles = {}, {}
    for offset, name in enumerate(ATTACK_NAMES):
        model = models[name]
        model.fit(x[train], y[train])
        scores = (
            model.predict_proba(x[holdout])[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(x[holdout])
        )
        predictions = model.predict(x[holdout])
        require_finite(scores, f"{name} scores")
        raw_auc = float(roc_auc_score(y[holdout], scores))
        bootstrap = document_group_bootstrap(
            y[holdout], scores, paired[holdout], reps, seed + offset
        )
        metrics[name] = {
            "raw_auc": raw_auc,
            "orientation_safe_auc": orientation_safe_auc(raw_auc),
            "orientation_safe_auc_document_group_ci95": _ci(bootstrap),
            "pr_auc": float(average_precision_score(y[holdout], scores)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y[holdout], predictions)
            ),
        }
        bundles[name] = {
            "y": y[holdout].copy(),
            "scores": np.asarray(scores),
            "groups": paired[holdout].copy(),
        }
    strongest = max(
        ATTACK_NAMES, key=lambda name: metrics[name]["orientation_safe_auc"]
    )
    return {
        "attacks": metrics,
        "strongest_attack": strongest,
        "strongest_orientation_safe_auc": metrics[strongest]["orientation_safe_auc"],
        "strongest_orientation_safe_auc_document_group_ci95": metrics[strongest][
            "orientation_safe_auc_document_group_ci95"
        ],
        "attack_train_documents": int(len(np.unique(paired[train]))),
        "attack_holdout_documents": int(len(np.unique(paired[holdout]))),
        "group_overlap": 0,
        "train_group_sha256": stable_hash(sorted(map(str, np.unique(paired[train])))),
        "holdout_group_sha256": stable_hash(
            sorted(map(str, np.unique(paired[holdout])))
        ),
    }, bundles


@torch.inference_mode()
def infer_safe_direct(
    model, rows: list[dict], device: torch.device, batch_size: int
) -> tuple[np.ndarray, dict]:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        count = int(row["leaves"].shape[0])
        if count not in (2, 3, 4):
            raise ValueError("safe arm requires one direct 2..4-child K4 merge")
        grouped[count].append((index, row))
    output = [None] * len(rows)
    angles, correction_abs = [], []
    geodesic_saturation = importance_saturation = importance_active = 0
    dtype = next(model.parameters()).dtype
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for count in (2, 3, 4):
        for start in range(0, len(grouped[count]), int(batch_size)):
            part = grouped[count][start : start + int(batch_size)]
            children = torch.stack([row["leaves"] for _, row in part]).to(
                device=device, dtype=dtype
            )
            masses = torch.stack([row["lengths"] for _, row in part]).to(
                device=device, dtype=dtype
            )
            if count < ARITY:
                children = F.pad(children, (0, 0, 0, ARITY - count))
                masses = F.pad(masses, (0, ARITY - count))
            roots = model.merge(children, masses)
            aux = model._last_merge_aux
            if not torch.isfinite(roots).all():
                raise FloatingPointError("safe direct roots are nonfinite")
            angles.append(aux["angle"].float().cpu().numpy())
            active, correction = aux["active_mask"], aux["corrections"]
            correction_abs.append(correction[active].abs().float().cpu().numpy())
            geodesic_saturation += int(aux["geodesic_saturated"].sum().item())
            importance_saturation += int(
                ((correction.abs() >= 0.95 * float(model.importance_beta)) & active)
                .sum()
                .item()
            )
            importance_active += int(active.sum().item())
            for (index, _), root in zip(part, roots.float().cpu()):
                output[index] = root.numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    roots = np.stack(output).astype(np.float32)
    require_finite(roots, "safe roots")
    angle_values, correction_values = (
        np.concatenate(angles),
        np.concatenate(correction_abs),
    )
    leaves = sum(int(row["leaves"].shape[0]) for row in rows)
    telemetry = {
        "direct_single_k4": True,
        "documents": len(rows),
        "leaves": leaves,
        "depth": {"minimum": 1, "maximum": 1, "mean": 1.0},
        "leaf_count_histogram": {
            str(count): len(grouped[count]) for count in (2, 3, 4)
        },
        "geodesic_angle_radians": _summary(angle_values),
        "geodesic_saturation_count": geodesic_saturation,
        "importance_correction_absolute": _summary(correction_values),
        "importance_active_slots": importance_active,
        "importance_saturation_count": importance_saturation,
        "seconds": elapsed,
        "documents_per_second": len(rows) / elapsed,
        "leaves_per_second": leaves / elapsed,
        "batch_size": int(batch_size),
        "device": str(device),
    }
    require_finite(
        [elapsed, telemetry["documents_per_second"], telemetry["leaves_per_second"]],
        "safe throughput",
    )
    return roots, telemetry


def _cosines(values: np.ndarray, targets: np.ndarray, label: str) -> np.ndarray:
    numerator = np.sum(values.astype(np.float64) * targets.astype(np.float64), axis=1)
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(targets, axis=1)
    result = numerator / denominator
    require_finite(result, label)
    return result


def group_mean_ci(values, groups, reps: int, seed: int) -> list[float]:
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups)
    if int(reps) != 2000 or len(unique) < 2:
        raise ValueError("mean CI requires frozen 2000 source-group bootstrap")
    by = {group: np.flatnonzero(groups == group) for group in unique}
    rng, estimates = np.random.default_rng(int(seed)), []
    for _ in range(int(reps)):
        sampled = rng.choice(unique, len(unique), replace=True)
        indexes = np.concatenate([by[group] for group in sampled])
        estimates.append(float(values[indexes].mean()))
    return _ci(estimates)


@torch.inference_mode()
def builtin_decoder_exact4(model, roots, rows, labels, groups, cfg, seed: int) -> dict:
    exact = np.asarray([int(row["leaves"].shape[0]) == ARITY for row in rows])
    if not exact.any():
        raise ValueError("safe arm has no exact-four rows")
    root_tensor = torch.from_numpy(roots[exact]).float()
    exact_rows = [row for row, keep in zip(rows, exact) if keep]
    selected_labels, selected_groups = labels[exact], groups[exact]
    device = next(model.parameters()).device
    prediction = []
    for start in range(0, len(root_tensor), int(cfg["batches"]["decoder"])):
        # Parent is the only decoder input: no lengths, labels, masks, IDs, or children.
        prediction.append(
            model.decode(
                root_tensor[start : start + int(cfg["batches"]["decoder"])].to(device)
            )
            .float()
            .cpu()
        )
    prediction = F.normalize(torch.cat(prediction), dim=-1)
    target = F.normalize(
        torch.stack([row["leaves"].float() for row in exact_rows]), dim=-1
    )
    result = {}
    masks = {"pooled": np.ones(len(exact_rows), dtype=bool)}
    masks.update({corpus: selected_labels == corpus for corpus in CORPUS_ORDER})
    for offset, scope in enumerate(SCOPES):
        indexes = np.flatnonzero(masks[scope])
        if not len(indexes):
            raise ValueError(f"missing exact-four decoder rows in {scope}")
        pred, children = prediction[indexes], target[indexes]
        flat_pred, flat_target = (
            pred.reshape(-1, pred.shape[-1]),
            children.reshape(-1, children.shape[-1]),
        )
        ranks = []
        for start in range(0, len(flat_pred), 512):
            score = flat_pred[start : start + 512] @ flat_target.T
            correct = torch.arange(start, min(start + 512, len(flat_pred)))
            target_score = score[torch.arange(len(score)), correct]
            candidate = torch.arange(len(flat_target))[None, :]
            ranks.append(
                (
                    1
                    + (
                        (score > target_score[:, None])
                        | (
                            (score == target_score[:, None])
                            & (candidate < correct[:, None])
                        )
                    ).sum(-1)
                ).cpu()
            )
        ranks = torch.cat(ranks).numpy()
        diagonal = F.cosine_similarity(pred, children, dim=-1)
        row_cosine = diagonal.mean(-1).numpy()
        row_r1 = (ranks.reshape(-1, ARITY) <= 1).mean(-1)
        result[scope] = {
            "rows": len(indexes),
            "children": ARITY * len(indexes),
            "input": "generated_parent_only",
            "oracle_inputs": [],
            "ordered_child_cosine": float(row_cosine.mean()),
            "ordered_child_cosine_ci95_group_bootstrap": group_mean_ci(
                row_cosine,
                selected_groups[indexes],
                2000,
                int(cfg["bootstrap"]["per_seed_seeds"][seed]) + 5000 + offset,
            ),
            "global_ordered_child_recall_at_1": float((ranks <= 1).mean()),
            "global_ordered_child_recall_at_5": float((ranks <= 5).mean()),
            "global_ordered_child_recall_at_10": float((ranks <= 10).mean()),
            "global_ordered_child_mrr": float((1.0 / ranks).mean()),
            "row_mean_recall_at_1_ci95_group_bootstrap": group_mean_ci(
                row_r1,
                selected_groups[indexes],
                2000,
                int(cfg["bootstrap"]["per_seed_seeds"][seed]) + 6000 + offset,
            ),
        }
    return {
        "protocol": "builtin_strict_decoder_parent_only_global_exact4_retrieval_v1",
        "exact_child_count": ARITY,
        "official_updates": 0,
        "scopes": result,
    }


def load_train_probe(entry: dict, cfg: dict, device: torch.device):
    path = ROOT / entry["candidate_probe"]
    require_hash(
        path,
        entry["candidate_probe_sha256"],
        f"seed {entry['seed']} frozen candidate TRAIN probe",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("arm") != "candidate"
        or checkpoint.get("fit_role") != "fresh_v6_train"
        or int(checkpoint.get("seed", -1)) != int(entry["seed"])
        or not isinstance(checkpoint.get("state_dict"), dict)
    ):
        raise ValueError("candidate probe is not the frozen seed-specific TRAIN fit")
    probe = ParentOnlyRetentionProbe(768, ARITY, int(cfg["train_probe"]["hidden"]))
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    return probe.to(device).eval(), {
        "path": entry["candidate_probe"],
        "sha256": entry["candidate_probe_sha256"],
        "fit_role": checkpoint["fit_role"],
        "arm": checkpoint["arm"],
        "seed": int(checkpoint["seed"]),
        "schedule_sha256": checkpoint["schedule_sha256"],
    }


@torch.inference_mode()
def frozen_train_retention(
    probe, roots, rows, labels, groups, entry, cfg, seed: int
) -> dict:
    exact = np.asarray([int(row["leaves"].shape[0]) == ARITY for row in rows])
    selected_roots = roots[exact]
    selected_rows = [row for row, keep in zip(rows, exact) if keep]
    selected_labels, selected_groups = labels[exact], groups[exact]
    predictions, device = [], next(probe.parameters()).device
    batch_size = int(cfg["train_probe"]["batch_size"])
    for start in range(0, len(selected_roots), batch_size):
        predictions.append(
            probe(
                torch.from_numpy(selected_roots[start : start + batch_size]).to(device)
            ).cpu()
        )
    predictions = F.normalize(torch.cat(predictions), dim=-1)
    children = F.normalize(
        torch.stack([row["leaves"].float() for row in selected_rows]), dim=-1
    )
    row_cosine = F.cosine_similarity(predictions, children, dim=-1).mean(-1).numpy()
    masks = {"pooled": np.ones(len(row_cosine), dtype=bool)}
    masks.update({corpus: selected_labels == corpus for corpus in CORPUS_ORDER})
    scopes = {}
    for offset, scope in enumerate(SCOPES):
        mask = masks[scope]
        if not mask.any():
            raise ValueError(f"missing frozen TRAIN-probe rows in {scope}")
        value = float(row_cosine[mask].mean())
        floor = float(entry["frozen_train_probe_floors"][scope])
        scopes[scope] = {
            "rows": int(mask.sum()),
            "ordered_child_cosine": value,
            "ordered_child_cosine_ci95_group_bootstrap": group_mean_ci(
                row_cosine[mask],
                selected_groups[mask],
                2000,
                int(cfg["bootstrap"]["per_seed_seeds"][seed]) + 20000 + offset,
            ),
            "frozen_train_only_floor": floor,
            "margin_above_floor": value - floor,
            "floor_pass": bool(value >= floor),
        }
    return {
        "protocol": cfg["train_probe"]["protocol"],
        "probe": entry["candidate_probe"],
        "probe_sha256": entry["candidate_probe_sha256"],
        "probe_fit_role": "fresh_v6_train",
        "official_updates": 0,
        "official_fitting": False,
        "exact_four_only": True,
        "input": "safe_direct_root",
        "all_scopes_floor_pass": all(value["floor_pass"] for value in scopes.values()),
        "scopes": scopes,
    }


def safe_scope_metrics(roots, targets, groups, labels, attack_cfg, cfg, seed: int):
    masks = {"pooled": np.ones(len(roots), dtype=bool)}
    masks.update({corpus: labels == corpus for corpus in CORPUS_ORDER})
    result, all_bundles = {}, {}
    for offset, scope in enumerate(SCOPES):
        indexes = np.flatnonzero(masks[scope])
        attacks, bundles = evaluate_fixed_attacks(
            targets[indexes],
            roots[indexes],
            groups[indexes],
            attack_cfg,
            int(cfg["bootstrap"]["reps"]),
            int(cfg["bootstrap"]["per_seed_seeds"][seed]) + 100 * offset,
        )
        cosine = _cosines(
            roots[indexes], targets[indexes], f"safe teacher cosine/{scope}"
        )
        result[scope] = {
            "documents": len(indexes),
            "teacher_cosine": float(cosine.mean()),
            "teacher_cosine_std": float(cosine.std()),
            "teacher_cosine_ci95_group_bootstrap": group_mean_ci(
                cosine,
                groups[indexes],
                2000,
                int(cfg["bootstrap"]["per_seed_seeds"][seed]) + 10000 + offset,
            ),
            "attacks": attacks,
            "health": _health(roots[indexes], cfg["collapse"], f"safe/{scope}"),
        }
        all_bundles[scope] = bundles
    return result, all_bundles


def expected_internal_merges(leaves: int) -> int:
    count, total = int(leaves), 0
    while count > 1:
        full, remainder = divmod(count, ARITY)
        merges = full + int(remainder >= 2)
        total += merges
        count = full + int(remainder > 0)
    return total


def _book_add(book: defaultdict, key, values) -> None:
    book[key].add(values)


def _book_result(book: dict) -> dict:
    return {
        str(key): stats.result()
        for key, stats in sorted(book.items(), key=lambda item: str(item[0]))
    }


@torch.inference_mode()
def reduce_full_document(
    model,
    shard: dict,
    corpus: str,
    device: torch.device,
    batch_size: int,
    telemetry: dict,
) -> torch.Tensor:
    nodes = shard["leaves"]
    masses = shard["lengths"].to(torch.float64)
    # False=original leaf, True=recursive child. Carried leaves remain leaves.
    recursive = torch.zeros(len(nodes), dtype=torch.bool)
    dtype = next(model.parameters()).dtype
    depth = 1
    observed_merges = 0
    while len(nodes) > 1:
        groups = [
            (start, min(ARITY, len(nodes) - start))
            for start in range(0, len(nodes), ARITY)
        ]
        merge_groups = [
            (index, start, width)
            for index, (start, width) in enumerate(groups)
            if width >= 2
        ]
        parents = {}
        for offset in range(0, len(merge_groups), int(batch_size)):
            part = merge_groups[offset : offset + int(batch_size)]
            child_batches, mass_batches, kind_batches = [], [], []
            for _, start, width in part:
                child_batches.append(
                    F.pad(
                        nodes[start : start + width].float(), (0, 0, 0, ARITY - width)
                    )
                )
                mass_batches.append(
                    F.pad(masses[start : start + width], (0, ARITY - width))
                )
                kind_batches.append(
                    F.pad(
                        recursive[start : start + width],
                        (0, ARITY - width),
                        value=False,
                    )
                )
            children = torch.stack(child_batches).to(device=device, dtype=dtype)
            child_masses = torch.stack(mass_batches).to(device=device, dtype=dtype)
            child_kinds = torch.stack(kind_batches)
            roots = model.merge(children, child_masses)
            aux = model._last_merge_aux
            predictions = model.decode(roots)  # parent only; no oracle arguments
            if not torch.isfinite(roots).all() or not torch.isfinite(predictions).all():
                raise FloatingPointError(
                    f"nonfinite full recursion/decoder output: {shard['id']}/depth{depth}"
                )
            active = aux["active_mask"]
            cosine = F.cosine_similarity(predictions, children, dim=-1)
            angles = aux["angle"].float().cpu().numpy()
            corrections = aux["corrections"].abs()
            weights = aux["effective_weights"]
            geo_sat = aux["geodesic_saturated"].cpu().numpy().astype(bool)
            imp_sat = (
                ((corrections >= 0.95 * float(model.importance_beta)) & active)
                .cpu()
                .numpy()
                .astype(bool)
            )
            for local, (group_index, start, width) in enumerate(part):
                parents[group_index] = (
                    roots[local].float().cpu(),
                    float(masses[start : start + width].sum()),
                )
                observed_merges += 1
                telemetry["merge_count"] += 1
                for key in (
                    "aggregate",
                    f"corpus:{corpus}",
                    f"depth:{depth}",
                    f"corpus:{corpus}|depth:{depth}",
                ):
                    _book_add(telemetry["geodesic"], key, [angles[local]])
                    telemetry["geodesic_saturation"][key] += int(geo_sat[local])
                    _book_add(
                        telemetry["importance"],
                        key,
                        corrections[local][active[local]].float().cpu().numpy(),
                    )
                    _book_add(
                        telemetry["effective_weight_max"],
                        key,
                        [float(weights[local][active[local]].max().item())],
                    )
                    telemetry["importance_active"][key] += int(
                        active[local].sum().item()
                    )
                    telemetry["importance_saturation"][key] += int(imp_sat[local].sum())
                active_cosine = cosine[local][active[local]].float().cpu().numpy()
                active_kinds = (
                    child_kinds[local][active[local].cpu()].numpy().astype(bool)
                )
                if len(active_cosine) != width:
                    raise AssertionError(
                        "full decoder did not score every active child"
                    )
                telemetry["decoded_merge_count"] += 1
                telemetry["decoded_child_count"] += width
                for child_cosine, is_recursive in zip(active_cosine, active_kinds):
                    kind = "recursive_child" if is_recursive else "leaf_child"
                    keys = (
                        "aggregate",
                        f"corpus:{corpus}",
                        f"depth:{depth}",
                        f"child_type:{kind}",
                        f"corpus:{corpus}|depth:{depth}",
                        f"corpus:{corpus}|child_type:{kind}",
                        f"depth:{depth}|child_type:{kind}",
                        f"corpus:{corpus}|depth:{depth}|child_type:{kind}",
                        f"active_arity:{width}",
                    )
                    for key in keys:
                        _book_add(telemetry["decoder_cosine"], key, [child_cosine])
        next_nodes, next_masses, next_recursive = [], [], []
        for group_index, (start, width) in enumerate(groups):
            if width == 1:
                next_nodes.append(nodes[start].float())
                next_masses.append(float(masses[start]))
                next_recursive.append(bool(recursive[start]))
            else:
                parent, mass = parents[group_index]
                next_nodes.append(parent)
                next_masses.append(mass)
                next_recursive.append(True)
        nodes = torch.stack(next_nodes)
        masses = torch.tensor(next_masses, dtype=torch.float64)
        recursive = torch.tensor(next_recursive, dtype=torch.bool)
        depth += 1
    expected = expected_internal_merges(int(shard["full_leaf_count"]))
    if observed_merges != expected or depth - 1 != int(shard["tree_depth"]):
        raise AssertionError("full recursion omitted an internal merge or depth")
    if int(round(float(masses[0]))) != int(shard["full_token_count"]):
        raise AssertionError("full recursion mass does not cover every token")
    return nodes[0]


@torch.inference_mode()
def infer_full_arm(
    model, full_by: dict[str, list], cfg: dict, device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict]:
    telemetry = {
        "merge_count": 0,
        "decoded_merge_count": 0,
        "decoded_child_count": 0,
        "geodesic": defaultdict(RunningStats),
        "importance": defaultdict(RunningStats),
        "effective_weight_max": defaultdict(RunningStats),
        "decoder_cosine": defaultdict(RunningStats),
        "geodesic_saturation": defaultdict(int),
        "importance_saturation": defaultdict(int),
        "importance_active": defaultdict(int),
    }
    roots, labels, tokens, leaves, depths = [], [], [], [], []
    expected_merges = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for corpus in CORPUS_ORDER:
        for indexed in full_by[corpus]:
            shard_path = ROOT / cfg["cache_dir"] / indexed["shard"]
            require_hash(
                shard_path,
                indexed["shard_sha256"],
                f"streamed full shard {indexed['id']}",
            )
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            expected = {
                key: indexed[key]
                for key in (
                    "id",
                    "source_index",
                    "source_document_id",
                    "corpus",
                    "full_token_count",
                    "full_leaf_count",
                    "tree_depth",
                )
            }
            validate_full_shard(shard, expected)
            root = reduce_full_document(
                model,
                shard,
                corpus,
                device,
                int(cfg["batches"]["full_merge"]),
                telemetry,
            )
            if not torch.isfinite(root).all():
                raise FloatingPointError("full document root is nonfinite")
            roots.append(root.numpy())
            labels.append(corpus)
            tokens.append(int(indexed["full_token_count"]))
            leaves.append(int(indexed["full_leaf_count"]))
            depths.append(int(indexed["tree_depth"]))
            expected_merges += expected_internal_merges(int(indexed["full_leaf_count"]))
            del shard
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    roots_array, labels_array = np.stack(roots).astype(np.float32), np.asarray(labels)
    require_finite(roots_array, "full roots")
    if (
        telemetry["merge_count"] != expected_merges
        or telemetry["decoded_merge_count"] != expected_merges
    ):
        raise AssertionError(
            "decoder/geodesic telemetry did not cover every internal merge"
        )
    if telemetry["geodesic_saturation"]["aggregate"] != sum(
        value
        for key, value in telemetry["geodesic_saturation"].items()
        if key.startswith("corpus:") and "|" not in key
    ):
        raise AssertionError("full geodesic corpus aggregation mismatch")
    distributions = {
        "token_count": {
            **_summary(tokens),
            "sum": int(sum(tokens)),
            "histogram": _histogram(tokens),
        },
        "leaf_count": {
            **_summary(leaves),
            "sum": int(sum(leaves)),
            "histogram": _histogram(leaves),
        },
        "tree_depth": {
            **_summary(depths),
            "sum": int(sum(depths)),
            "histogram": _histogram(depths),
        },
        "by_corpus": {},
    }
    for corpus in CORPUS_ORDER:
        mask = labels_array == corpus
        distributions["by_corpus"][corpus] = {
            "token_count": _summary(np.asarray(tokens)[mask]),
            "leaf_count": _summary(np.asarray(leaves)[mask]),
            "tree_depth": {
                **_summary(np.asarray(depths)[mask]),
                "histogram": _histogram(np.asarray(depths)[mask]),
            },
        }
    geometry = {}
    for key in telemetry["geodesic"]:
        geometry[str(key)] = {
            "merge_count": telemetry["geodesic"][key].count,
            "geodesic_angle_radians": telemetry["geodesic"][key].result(),
            "geodesic_saturation_count": int(telemetry["geodesic_saturation"][key]),
            "importance_correction_absolute": telemetry["importance"][key].result(),
            "maximum_effective_child_weight": telemetry["effective_weight_max"][
                key
            ].result(),
            "importance_active_slots": int(telemetry["importance_active"][key]),
            "importance_saturation_count": int(telemetry["importance_saturation"][key]),
        }
    decoder = _book_result(telemetry["decoder_cosine"])
    throughput = {
        "documents": len(roots),
        "tokens": int(sum(tokens)),
        "leaves": int(sum(leaves)),
        "internal_merges": expected_merges,
        "seconds": elapsed,
        "documents_per_second": len(roots) / elapsed,
        "tokens_per_second": sum(tokens) / elapsed,
        "leaves_per_second": sum(leaves) / elapsed,
        "internal_merges_per_second": expected_merges / elapsed,
        "merge_batch_size": int(cfg["batches"]["full_merge"]),
        "device": str(device),
    }
    require_finite(list(throughput.values())[4:9], "full throughput")
    return (
        roots_array,
        labels_array,
        {
            "protocol": "unrestricted_every_token_contiguous_k4_recursion_v1",
            "every_available_token_processed": True,
            "token_cap": None,
            "maximum_leaves": None,
            "recursion_depth_cap": None,
            "truncated_documents": 0,
            "full_root_teacher_target": False,
            "teacher_comparison_metrics": [],
            "attacks": [],
            "document_distributions": distributions,
            "all_level_geometry_and_saturation": geometry,
            "builtin_decoder_active_slot_child_cosine": {
                "protocol": "strict_parent_only_decoder_every_internal_merge_active_slots_v1",
                "decoder_inputs": ["generated_parent"],
                "oracle_inputs": [],
                "official_updates": 0,
                "expected_internal_merges": expected_merges,
                "observed_internal_merges": telemetry["decoded_merge_count"],
                "active_children_scored": telemetry["decoded_child_count"],
                "aggregations": decoder,
            },
            "throughput": throughput,
        },
    )


def full_health(roots, labels, cfg) -> dict:
    masks = {"pooled": np.ones(len(roots), dtype=bool)}
    masks.update({corpus: labels == corpus for corpus in CORPUS_ORDER})
    return {
        scope: _health(roots[masks[scope]], cfg["collapse"], f"full/{scope}")
        for scope in SCOPES
    }


def three_seed_shared_ci(
    primary_bundles: list[dict], reps: int, seed: int
) -> tuple[float, list[float]]:
    if len(primary_bundles) != 3 or int(reps) != 2000:
        raise ValueError("primary aggregate requires three seeds and 2000 reps")
    reference = primary_bundles[0]
    for bundle in primary_bundles[1:]:
        if not np.array_equal(bundle["y"], reference["y"]) or not np.array_equal(
            bundle["groups"], reference["groups"]
        ):
            raise AssertionError(
                "three-seed labels/heldout source groups are not shared"
            )
    unique = np.unique(reference["groups"])
    members = {group: np.flatnonzero(reference["groups"] == group) for group in unique}
    rng, estimates = np.random.default_rng(int(seed)), []
    for _ in range(int(reps)):
        sampled = rng.choice(unique, len(unique), replace=True)
        indexes = np.concatenate([members[group] for group in sampled])
        estimates.append(
            float(
                np.mean(
                    [
                        orientation_safe_auc(
                            float(
                                roc_auc_score(
                                    bundle["y"][indexes], bundle["scores"][indexes]
                                )
                            )
                        )
                        for bundle in primary_bundles
                    ]
                )
            )
        )
    point = float(
        np.mean(
            [
                orientation_safe_auc(
                    float(roc_auc_score(bundle["y"], bundle["scores"]))
                )
                for bundle in primary_bundles
            ]
        )
    )
    return point, _ci(estimates)


def _full_arm_complete(full: dict) -> bool:
    decoder = full["builtin_decoder_active_slot_child_cosine"]
    aggregations = decoder["aggregations"]
    required = {"aggregate"}
    for corpus in CORPUS_ORDER:
        required.update(
            {
                f"corpus:{corpus}",
                f"corpus:{corpus}|child_type:leaf_child",
                f"corpus:{corpus}|child_type:recursive_child",
            }
        )
    return bool(
        full["every_available_token_processed"] is True
        and full["truncated_documents"] == 0
        and full["token_cap"] is None
        and full["maximum_leaves"] is None
        and full["recursion_depth_cap"] is None
        and full["full_root_teacher_target"] is False
        and full["teacher_comparison_metrics"] == []
        and full["attacks"] == []
        and full["throughput"]["documents"] == len(CORPUS_ORDER) * TARGET_DOCUMENTS
        and decoder["expected_internal_merges"] == decoder["observed_internal_merges"]
        and required.issubset(aggregations)
        and any(key.startswith("depth:") for key in aggregations)
        and set(full["document_distributions"]["by_corpus"]) == set(CORPUS_ORDER)
    )


def reporting_gate(seed_results: list[dict], mean_auc: float, cfg: dict) -> dict:
    gate = cfg["reporting_gate"]
    sole_target = float(gate["sole_auc_target"])
    seed_guard = sole_target + float(gate["per_seed_guard_margin"])
    pooled_cosines = [
        entry["safe_arm"]["scopes"]["pooled"]["teacher_cosine"]
        for entry in seed_results
    ]
    per_seed_auc = [
        entry["safe_arm"]["scopes"]["pooled"]["attacks"][
            "strongest_orientation_safe_auc"
        ]
        for entry in seed_results
    ]
    builtin = all(
        entry["safe_arm"]["builtin_decoder_exact4"]["scopes"][scope][
            "ordered_child_cosine"
        ]
        >= float(
            gate["every_seed_every_safe_scope_builtin_exact4_child_cosine_at_least"]
        )
        for entry in seed_results
        for scope in SCOPES
    )
    train_probe = all(
        entry["safe_arm"]["frozen_v6_train_probe_retention"]["scopes"][scope][
            "floor_pass"
        ]
        for entry in seed_results
        for scope in SCOPES
    )
    healthy = True
    full_complete = len(seed_results) == len(SEED_ORDER)
    for entry in seed_results:
        safe = entry["safe_arm"]
        healthy &= safe["telemetry"]["geodesic_saturation_count"] == 0
        healthy &= safe["telemetry"]["importance_saturation_count"] == 0
        healthy &= all(
            value["all_finite"]
            and not value["collapse"]
            and value["output_norm_saturation_count"] == 0
            for value in (scope["health"] for scope in safe["scopes"].values())
        )
        full = entry["full_document_arm"]
        aggregate = full["all_level_geometry_and_saturation"]["aggregate"]
        healthy &= (
            aggregate["geodesic_saturation_count"] == 0
            and aggregate["importance_saturation_count"] == 0
        )
        healthy &= all(
            value["all_finite"]
            and not value["collapse"]
            and value["output_norm_saturation_count"] == 0
            for value in full["health"].values()
        )
        full_complete &= _full_arm_complete(full)
    components = {
        "three_seed_mean_strongest_pooled_point_below_sole_auc_target": bool(
            mean_auc < sole_target
        ),
        "every_seed_strongest_pooled_point_below_derived_guard": bool(
            all(value < seed_guard for value in per_seed_auc)
        ),
        "every_seed_pooled_teacher_cosine_strictly_above_required_value": bool(
            all(
                value > float(gate["every_seed_pooled_teacher_cosine_strictly_above"])
                for value in pooled_cosines
            )
        ),
        "every_seed_every_safe_scope_builtin_exact4_child_cosine_at_least_required_value": bool(
            builtin
        ),
        "every_seed_every_frozen_train_probe_floor_pass": bool(train_probe),
        "no_safe_or_full_collapse_nonfinite_or_saturation": bool(healthy),
        "full_arm_mandatory_and_complete": bool(full_complete),
    }
    return {
        "reporting_only": True,
        "suppresses_or_selects_outcomes": False,
        "sole_auc_target": sole_target,
        "per_seed_guard_derived_from_target_and_margin": seed_guard,
        "mean_pooled_teacher_cosine": float(np.mean(pooled_cosines)),
        "minimum_seed_pooled_teacher_cosine": float(min(pooled_cosines)),
        "components": components,
        "passed": bool(all(components.values())),
    }


def _evaluation_population(safe_by: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    rows = [row for corpus in CORPUS_ORDER for row in safe_by[corpus]]
    groups = np.asarray([str(row["id"]) for row in rows])
    labels = np.asarray(
        [corpus for corpus in CORPUS_ORDER for _ in safe_by[corpus]]
    )
    targets = np.stack(
        [row["teacher_target"].float().numpy() for row in rows]
    ).astype(np.float32)
    if len(rows) != 4 * TARGET_DOCUMENTS or len(set(groups)) != len(groups):
        raise ValueError("pooled safe source groups are missing/duplicated")
    require_finite(targets, "safe teacher targets")
    return rows, groups, labels, targets


def _fixed_attack_config(cfg: dict) -> dict:
    attack_cfg = yaml.safe_load((ROOT / cfg["attack_config"]).read_text())
    if tuple(attack_cfg) != (
        "protocol_version",
        "split_seed",
        "model_seed",
        "test_size",
        "logistic_c",
        "logistic_max_iter",
        "rbf_c",
        "rbf_gamma",
    ):
        raise ValueError("fixed logistic/RBF attack config changed")
    return attack_cfg


def _replay_device(device_name: str) -> torch.device:
    if device_name not in ("cpu", "cuda"):
        raise ValueError("replay device must be explicitly cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for replay but is unavailable")
    return torch.device(device_name)


def _evaluation_model(entry: dict, seed: int, device: torch.device, *, replay: bool):
    require_hash(
        ROOT / entry["checkpoint"],
        entry["checkpoint_sha256"],
        f"seed {seed} AE checkpoint",
    )
    if replay:
        # The validation builder rejects "official"/"test" in its path string.
        # Check the configured artifact's relative path, not an unrelated parent
        # directory name, while still loading exactly the hash-verified file.
        original_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            model, checkpoint = build_model(Path(entry["checkpoint"]), device)
        finally:
            os.chdir(original_directory)
    else:
        model, checkpoint = build_model(ROOT / entry["checkpoint"], device)
    if (
        checkpoint.get("variant") != "autoencoder"
        or model.arity != ARITY
        or model.strict_decoder is None
        or model.decoder_mode != "strict"
    ):
        raise ValueError(f"seed {seed} is not a decoder-bearing strict-K4 AE")
    checkpoint_seed = int(
        checkpoint.get("seed", checkpoint.get("config", {}).get("seed", -1))
    )
    if checkpoint_seed != seed:
        raise ValueError(f"seed {seed} checkpoint seed mismatch")
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError(f"seed {seed} model parameters are nonfinite")
    return model, checkpoint


def _claim_replay_output() -> Path:
    directory = ROOT / "replays"
    if directory.is_symlink():
        raise ValueError("replays namespace must not be a symbolic link")
    directory.mkdir(exist_ok=True)
    output = directory / "official_v6"
    output.mkdir()
    fsync_directory(directory)
    return output


def _difference(historical, replay) -> dict:
    return {
        "historical": historical,
        "replay": replay,
        "replay_minus_historical": replay - historical,
    }


def _compare_historical_results(result: dict) -> dict:
    historical = _historical_result()
    old_seeds = {int(entry["seed"]): entry for entry in historical["seeds"]}
    new_seeds = {int(entry["seed"]): entry for entry in result["seeds"]}
    if set(old_seeds) != set(SEED_ORDER) or set(new_seeds) != set(SEED_ORDER):
        raise ValueError("cannot compare official results with incomplete seed sets")
    comparison = {
        "historical_result": {
            "path": HISTORICAL_RESULT_PATH,
            "sha256": HISTORICAL_RESULT_SHA256,
        },
        "difference_definition": "replay minus historical; no tolerance or selection",
        "three_seed_scope_aggregates": {},
        "seeds": {},
        "reporting_gate": {
            "historical": historical["reporting_only_gate"],
            "replay": result["reporting_only_gate"],
        },
    }
    for scope in SCOPES:
        old = historical["three_seed_scope_aggregates"][scope]
        new = result["three_seed_scope_aggregates"][scope]
        comparison["three_seed_scope_aggregates"][scope] = {
            "mean_auc": _difference(old["mean_auc"], new["mean_auc"]),
            "ci95": [
                _difference(old_value, new_value)
                for old_value, new_value in zip(old["ci95"], new["ci95"])
            ],
        }
    for seed in SEED_ORDER:
        old, new = old_seeds[seed], new_seeds[seed]
        old_safe, new_safe = old["safe_arm"], new["safe_arm"]
        seed_comparison = {"safe_scopes": {}, "full_document": {}}
        for scope in SCOPES:
            old_scope, new_scope = old_safe["scopes"][scope], new_safe["scopes"][scope]
            old_attacks, new_attacks = old_scope["attacks"], new_scope["attacks"]
            scope_comparison = {
                "teacher_cosine": _difference(
                    old_scope["teacher_cosine"], new_scope["teacher_cosine"]
                ),
                "strongest_attack": {
                    "historical": old_attacks["strongest_attack"],
                    "replay": new_attacks["strongest_attack"],
                },
                "strongest_orientation_safe_auc": _difference(
                    old_attacks["strongest_orientation_safe_auc"],
                    new_attacks["strongest_orientation_safe_auc"],
                ),
                "fixed_attacks_orientation_safe_auc": {
                    attack: _difference(
                        old_attacks["attacks"][attack]["orientation_safe_auc"],
                        new_attacks["attacks"][attack]["orientation_safe_auc"],
                    )
                    for attack in ATTACK_NAMES
                },
                "holdout_group_sha256": {
                    "historical": old_attacks["holdout_group_sha256"],
                    "replay": new_attacks["holdout_group_sha256"],
                },
            }
            for label, field in (
                ("builtin_decoder_cosine", "builtin_decoder_exact4"),
                ("frozen_train_probe_cosine", "frozen_v6_train_probe_retention"),
            ):
                scope_comparison[label] = _difference(
                    old_safe[field]["scopes"][scope]["ordered_child_cosine"],
                    new_safe[field]["scopes"][scope]["ordered_child_cosine"],
                )
            seed_comparison["safe_scopes"][scope] = scope_comparison
        old_full, new_full = old["full_document_arm"], new["full_document_arm"]
        for field in ("documents", "tokens", "leaves", "internal_merges"):
            seed_comparison["full_document"][field] = _difference(
                old_full["throughput"][field], new_full["throughput"][field]
            )
        old_decoder = old_full["builtin_decoder_active_slot_child_cosine"]
        new_decoder = new_full["builtin_decoder_active_slot_child_cosine"]
        seed_comparison["full_document"]["active_child_cosine"] = _difference(
            old_decoder["aggregations"]["aggregate"]["mean"],
            new_decoder["aggregations"]["aggregate"]["mean"],
        )
        comparison["seeds"][str(seed)] = seed_comparison
    return comparison


def render_report(result: dict) -> str:
    replay = "replay" in result
    lines = [
        "# Strict-K4 AUTOENCODER official test v6 cached replay"
        if replay
        else "# Strict-K4 AUTOENCODER official test v6",
        "",
        "This is a new evaluation of the published official cache, not the original one-shot collection or an untouched new population. It reuses all 4000 paired safe/full documents and three frozen checkpoints, fits the declared fixed attacks, and performs no model or probe updates. Historical outputs remain separate."
        if replay
        else "This atomic one-shot evaluates the three confirmed decoder-bearing AE checkpoints on one untouched test population. The same first 1000 safe-prefix-length-eligible source documents per corpus feed both mandatory arms. No post-access tuning, selection, rerun, suppression, or updates occur.",
        "",
        "The sole AUC target is the frozen strict bound `<0.60`; the per-seed guard is derived from that target and its declared margin.",
        "",
        "## Safe teacher-comparable arm",
        "",
        "| seed | strongest pooled attack | safe AUC | 95% source-group CI | pooled cosine | macro cosine | built-in exact4 pooled cosine | TRAIN probe/floor |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in result["seeds"]:
        pooled = entry["safe_arm"]["scopes"]["pooled"]["attacks"]
        ci = pooled["strongest_orientation_safe_auc_document_group_ci95"]
        retention = entry["safe_arm"]["frozen_v6_train_probe_retention"]["scopes"][
            "pooled"
        ]
        lines.append(
            f"| {entry['seed']} | {pooled['strongest_attack']} | {pooled['strongest_orientation_safe_auc']:.6f} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] | {entry['safe_arm']['scopes']['pooled']['teacher_cosine']:.6f} | "
            f"{entry['safe_arm']['macro_teacher_cosine']:.6f} | "
            f"{entry['safe_arm']['builtin_decoder_exact4']['scopes']['pooled']['ordered_child_cosine']:.6f} | "
            f"{retention['ordered_child_cosine']:.6f}/{retention['frozen_train_only_floor']:.6f} |"
        )
    primary = result["primary_three_seed_aggregate"]
    lines += [
        "",
        f"Three-seed mean strongest pooled point: **{primary['mean_auc']:.6f}**; shared-heldout-group 2000-bootstrap 95% CI `[{primary['ci95'][0]:.6f}, {primary['ci95'][1]:.6f}]`.",
        "",
        "Pooled and per-corpus three-seed shared-group aggregate points and CIs are retained in `results.json`.",
        "",
    ]
    lines += [
        "## Unrestricted full-document recursion arm",
        "",
        "Every available token is represented by an exact contiguous <=1024-token leaf; there is no token, leaf, or depth cap and no truncation. This arm has no full-root teacher target, AUC, or teacher cosine.",
        "",
        "| seed | tokens | leaves | max tokens/doc | max leaves/doc | max depth | merges | decoder active-child cosine | full root rank | docs/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in result["seeds"]:
        full = entry["full_document_arm"]
        dist = full["document_distributions"]
        throughput = full["throughput"]
        lines.append(
            f"| {entry['seed']} | {int(dist['token_count']['sum'])} | {int(dist['leaf_count']['sum'])} | {int(dist['token_count']['maximum'])} | {int(dist['leaf_count']['maximum'])} | {int(dist['tree_depth']['maximum'])} | {throughput['internal_merges']} | {full['builtin_decoder_active_slot_child_cosine']['aggregations']['aggregate']['mean']:.6f} | {full['health']['pooled']['centered_numeric_rank']} | {throughput['documents_per_second']:.3f} |"
        )
    lines += [
        "",
        "Full per-corpus/depth/leaf-vs-recursive-child decoder cosine, every-depth geodesic/importance saturation, token/leaf/depth distributions, maxima, health, and throughput are retained in `results.json`.",
        "",
        "## Reporting-only gate",
        "",
    ]
    for name, value in result["reporting_only_gate"]["components"].items():
        lines.append(f"- `{name}`: **{value}**")
    lines += [
        f"- conjunction: **{result['reporting_only_gate']['passed']}** (reporting only).",
        "",
        "## Provenance",
        "",
        "Checkpoint and source hashes, actual `variant=autoencoder` and strict-decoder state proof, genuine nonzero encoder/decoder change, seed-specific fresh-v6-TRAIN probe identities and frozen TRAIN-only floors, exact source counts and hashes, test role, every cache-shard hash, environment, and timing are in `results.json`.",
        "",
    ]
    if replay:
        point = result["historical_comparison"]["three_seed_scope_aggregates"]["pooled"]["mean_auc"]
        lines += [
            "## Comparison with the retained historical result",
            "",
            f"Historical pooled mean AUC: `{point['historical']:.17g}`; replay: `{point['replay']:.17g}`; replay minus historical: `{point['replay_minus_historical']:.17g}`.",
            "",
            "Per-seed and per-corpus attack, cosine, population, and aggregate CI differences are recorded in `results.json`. No tolerance, retuning, or suppression is applied. Different runtime or device numerics need not reproduce the historical bytes.",
            "",
            "The cache manifest and completion marker are explicitly identified publication derivatives. Historical and normalized configuration, development, freeze, and cache identities are recorded separately; the original hashes are not rebound.",
            "",
        ]
    return "\n".join(lines)


def evaluate(
    config_path: Path = CONFIG, device_name: str = "cuda", *, replay: bool = False
) -> Path:
    cfg = load_config(config_path)
    if replay:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        _pinned_package_input(REPLAY_CONFIG_PATH, REPLAY_CONFIG_SHA256)
        device = _replay_device(device_name)
    # Must precede cache access and one-shot evaluation claim.
    immutable = validate_immutable_provenance(cfg, check_checkpoint_tensors=True)
    if replay:
        from public_replay import REPLAY_ID

        safe_by, full_by, cache_provenance = load_cache(
            cfg, replay_immutable=immutable
        )
        rows, groups, labels, targets = _evaluation_population(safe_by)
        attack_cfg = _fixed_attack_config(cfg)
        for entry in cfg["seeds"]:
            probe, _ = load_train_probe(entry, cfg, torch.device("cpu"))
            del probe
        required_free = int(cfg["storage"]["result_minimum_free_bytes"])
        if shutil.disk_usage(ROOT).free < required_free:
            raise RuntimeError("insufficient storage for official replay results")
        output = _claim_replay_output()
        staging = output
        atomic_text(
            output / "STARTED.json",
            json.dumps(
                {
                    "replay_id": REPLAY_ID,
                    "status": "started_not_complete",
                    "population": "published_official_cache",
                    "historical_output_overwrite": False,
                    "new_untouched_test_collection": False,
                    "public_contract": cache_provenance["public_replay"],
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
        )
    else:
        output = ROOT / cfg["output_dir"]
        failed_marker = output.with_name(output.name + ".EVALUATION_STARTED")
        if not failed_marker.is_file() or output.exists():
            raise FileExistsError(
                "recovery requires the spent initial evaluation marker and no finalized output"
            )
        marker = output.with_name(output.name + ".EVALUATION_RECOVERY_STARTED")
        claim_once(marker, (output,))
        staging = output.with_name(output.name + f".staging.{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
    started_wall = time.time()
    try:
        if not replay:
            safe_by, full_by, cache_provenance = load_cache(cfg)
            rows, groups, labels, targets = _evaluation_population(safe_by)
            attack_cfg = _fixed_attack_config(cfg)
            device = torch.device(
                device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
            )
        configured = {int(entry["seed"]): entry for entry in cfg["seeds"]}
        seed_results = []
        shared_bundles = {scope: [] for scope in SCOPES}
        for seed in SEED_ORDER:
            entry = configured[seed]
            model, checkpoint = _evaluation_model(entry, seed, device, replay=replay)
            safe_roots, safe_telemetry = infer_safe_direct(
                model, rows, device, int(cfg["batches"]["safe_root"])
            )
            scopes, bundles = safe_scope_metrics(
                safe_roots, targets, groups, labels, attack_cfg, cfg, seed
            )
            for scope in SCOPES:
                strongest = scopes[scope]["attacks"]["strongest_attack"]
                shared_bundles[scope].append(bundles[scope][strongest])
            builtin = builtin_decoder_exact4(
                model, safe_roots, rows, labels, groups, cfg, seed
            )
            probe, probe_metadata = load_train_probe(entry, cfg, device)
            retention = frozen_train_retention(
                probe, safe_roots, rows, labels, groups, entry, cfg, seed
            )
            full_roots, full_labels, full = infer_full_arm(model, full_by, cfg, device)
            full["health"] = full_health(full_roots, full_labels, cfg)
            seed_results.append(
                {
                    "seed": seed,
                    "checkpoint": entry["checkpoint"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "actual_autoencoder_proof": immutable["checkpoints"][seed],
                    "identical_source_order_sha256": cache_provenance[
                        "pooled_document_order_sha256"
                    ],
                    (
                        "model_or_probe_updates_or_selection"
                        if replay
                        else "official_fits_updates_or_selection"
                    ): 0,
                    "safe_arm": {
                        "protocol": "teacher_comparable_direct_k4_safe_prefix_v1",
                        "prefix_tokens": [1025, 4096],
                        "teacher_target_max_tokens": 4096,
                        "one_direct_root_per_document": True,
                        "macro_teacher_cosine": float(
                            np.mean(
                                [
                                    scopes[corpus]["teacher_cosine"]
                                    for corpus in CORPUS_ORDER
                                ]
                            )
                        ),
                        "scopes": scopes,
                        "telemetry": safe_telemetry,
                        "builtin_decoder_exact4": builtin,
                        "frozen_v6_train_probe_retention": retention,
                        "frozen_v6_train_probe_metadata": probe_metadata,
                    },
                    "full_document_arm": full,
                }
            )
            del model, checkpoint, probe, full_roots, safe_roots
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scope_aggregates = {}
        for scope in SCOPES:
            point, ci95 = three_seed_shared_ci(
                shared_bundles[scope],
                int(cfg["bootstrap"]["reps"]),
                int(cfg["bootstrap"]["three_seed_shared_seeds"][scope]),
            )
            scope_aggregates[scope] = {
                "definition": "mean of each seed's strongest orientation-safe fixed attack in this scope",
                "mean_auc": point,
                "ci95": ci95,
                "bootstrap_unit": "shared held-out source-document group",
                "shared_groups_verified": True,
                "reps": 2000,
                "seed": int(cfg["bootstrap"]["three_seed_shared_seeds"][scope]),
            }
        primary = scope_aggregates["pooled"]
        gate = reporting_gate(seed_results, primary["mean_auc"], cfg)
        result = {
            "stage": CACHE_NAMESPACE,
            "protocol_version": 6,
            "official_test_evaluated": True,
            "test_role_only": True,
            "report_all_outcomes": True,
            "same_accepted_documents_both_arms": True,
            "seed_order": list(SEED_ORDER),
            "corpus_order": list(CORPUS_ORDER),
            "declared_scan_window_per_corpus": list(SCAN),
            "prior_ranges_and_buffer_refused": {
                key: list(value) for key, value in PRIOR_RANGES.items()
            },
            "selection": "deterministic first 1000 eligible by safe-prefix length only",
            "immutable_provenance": immutable,
            "cache_provenance": cache_provenance,
            "attack_config": cfg["attack_config"],
            "attack_config_sha256": cfg["attack_config_sha256"],
            "bootstrap": cfg["bootstrap"],
            "seeds": seed_results,
            "three_seed_scope_aggregates": scope_aggregates,
            "primary_three_seed_aggregate": primary,
            "reporting_only_gate": gate,
            "sole_auc_target": cfg["reporting_gate"]["sole_auc_target"],
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "device_requested": device_name,
                "device_used": str(device),
                "cuda_available": torch.cuda.is_available(),
            },
            "started_unix": started_wall,
            "completed_unix": time.time(),
        }
        if replay:
            import scipy
            import sklearn

            result["replay"] = {
                "replay_id": REPLAY_ID,
                "output_namespace": "replays/official_v6",
                "execution_kind": "rerun_of_published_official_cache",
                "new_untouched_test_collection": False,
                "original_collection_evaluation_ordinal": None,
                "frozen_before_original_official_access_claimed_for_replay": False,
                "historical_outputs_overwritten": False,
                "model_or_probe_training_updates": 0,
                "fixed_attacks_refit": True,
                "cached_population_documents_per_arm": len(rows),
                "cache_release": cache_provenance["public_replay"],
                "execution_source_sha256": {
                    relative: sha256_file(ROOT / relative)
                    for relative in (
                        "scripts/replay_official_v6.py",
                        "scripts/public_replay.py",
                        "scripts/official_v6/evaluate_official_test_v6_recovery.py",
                        "scripts/official_v6/official_test_v6_protocol.py",
                        "scripts/official_v6/prepare_official_test_v6.py",
                        "src/recursive_geodesic_composition/architecture.py",
                        "src/recursive_geodesic_composition/evaluate_kary_validation.py",
                        "src/recursive_geodesic_composition/validation_output_calibration.py",
                    )
                },
            }
            result["environment"].update(
                {
                    "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                    "pyyaml": yaml.__version__,
                    "torch_cpu_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "thread_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                        )
                    },
                }
            )
            result["historical_comparison"] = _compare_historical_results(result)
        else:
            result.update(
                {
                    "frozen_before_access": True,
                    "atomic_one_shot": True,
                    "evaluator_recovery": {
                        "initial_metrics_produced": False,
                        "initial_failure": "KeyError: 'confirmation'",
                        "reused_prepared_cache_without_source_rescan": True,
                        "recovery_scope": "replace stale v5 provenance-key lookup with frozen clean-v6 provenance-key lookup",
                    },
                    "post_access_tuning_selection_rerun_or_suppression": False,
                }
            )
        require_storage(
            staging,
            int(cfg["storage"]["result_minimum_free_bytes"]),
            "atomic v6 result publication",
        )
        result_path = staging / "results.json"
        atomic_text(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
        atomic_text(staging / "REPORT.md", render_report(result))
        hashes = [
            f"{sha256_file(result_path)}  results.json",
            f"{sha256_file(staging / 'REPORT.md')}  REPORT.md",
            *(
                [f"{sha256_file(staging / 'STARTED.json')}  STARTED.json"]
                if replay
                else []
            ),
        ]
        atomic_text(staging / "SHA256SUMS", "\n".join(hashes) + "\n")
        atomic_text(staging / "DONE", f"{sha256_file(result_path)}  results.json\n")
        fsync_directory(staging)
        if not replay:
            os.replace(staging, output)
        fsync_directory(output.parent)
        return output
    except BaseException:
        if replay:
            (output / "DONE").unlink(missing_ok=True)
            atomic_text(
                output / "FAILED.json",
                json.dumps(
                    {
                        "replay_id": REPLAY_ID,
                        "status": "failed_not_complete",
                        "failure_type": type(sys.exception()).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
            )
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="evaluate the pinned anonymous cache in create-only replays/official_v6",
    )
    args = parser.parse_args()
    evaluate(args.config, args.device, replay=args.replay)


if __name__ == "__main__":
    main()
