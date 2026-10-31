"""Validation-only, document-grouped information-retention probes.

This module deliberately accepts train/validation multicorpus caches only.  It has
no test-role CLI or path discovery and never imports the attack evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import nn

from .architecture import GeodesicMerge, weighted_average

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")
ARMS = ("pooling", "encoder", "autoencoder")
FORBIDDEN_ROLE_WORDS = ("test", "official", "attack")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_document_id(row_id: str) -> str:
    """Rows from augmentations share the exact corpus-prefixed source ID."""
    return str(row_id)


def reject_forbidden_path(path: str | Path) -> None:
    parts = [part.lower() for part in Path(path).parts]
    if any(any(word in part for word in FORBIDDEN_ROLE_WORDS) for part in parts):
        raise ValueError(f"refusing forbidden test/official/attack path: {path}")


def verify_manifest(root: Path, entry: dict) -> dict:
    checkpoint = root / entry["checkpoint"]
    manifest_path = root / entry["manifest"]
    # Frozen result names are allowed; only role/range paths are forbidden.  The
    # two paths are exact protocol constants rather than discovered artifacts.
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing frozen checkpoint/manifest: {checkpoint}, {manifest_path}"
        )
    actual = sha256_file(checkpoint)
    if actual != entry["sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch for {checkpoint}: {actual}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("official_test_evaluated") is not False:
        raise ValueError(f"manifest is not explicitly validation-only: {manifest_path}")
    if manifest.get("checkpoint_sha256") != actual:
        raise ValueError(f"manifest checkpoint hash mismatch: {manifest_path}")
    verified = {
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": actual,
        "manifest": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256_file(manifest_path),
    }
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("validation_log", "validation_log_sha256"),
        ("run_status", "run_status_sha256"),
    ):
        relative = manifest.get(path_key)
        expected = manifest.get(hash_key)
        if not relative or not expected:
            raise ValueError(f"manifest lacks {path_key}/{hash_key}: {manifest_path}")
        artifact = root / relative
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"manifest artifact hash mismatch: {artifact}")
        verified[hash_key] = expected
    return verified


def load_protocol(root: Path, config_path: Path) -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load(config_path.read_text())
    if (
        cfg.get("validation_only") is not True
        or cfg.get("official_test_evaluated") is not False
    ):
        raise ValueError("retention config must be explicitly validation-only")
    if any(
        word in str(cfg.get("cache_dir", "")).lower() for word in FORBIDDEN_ROLE_WORDS
    ):
        raise ValueError("cache path may not name a forbidden role")
    source = yaml.safe_load((root / cfg["source_config"]).read_text())
    if "test" in source or any("test" in str(k).lower() for k in source):
        raise ValueError("source config unexpectedly declares a test role")
    if tuple(c["id"] for c in source["corpora"]) != CORPORA:
        raise ValueError("pinned corpus order changed")
    for corpus in source["corpora"]:
        if int(corpus["train_start"]) != 0 or int(corpus["train_documents"]) != 8000:
            raise ValueError("pinned training range changed")
        if (
            int(corpus["validation_start"]) != 12000
            or int(corpus["validation_documents"]) != 1000
        ):
            raise ValueError("pinned validation range changed")
    verified = {
        name: verify_manifest(root, cfg["frozen"][name])
        for name in ("encoder", "autoencoder")
    }
    return cfg, source, verified


def load_caches(root: Path, cfg: dict, source: dict) -> tuple[dict, dict, dict]:
    train, validation, sidecars = {}, {}, {}
    source_by = {c["id"]: c for c in source["corpora"]}
    for corpus in CORPORA:
        train_path = root / cfg["cache_dir"] / corpus / "train.pt"
        val_path = root / cfg["cache_dir"] / corpus / "validation.pt"
        # Exact role filenames, never globbed/discovered.
        train[corpus] = torch.load(train_path, map_location="cpu", weights_only=False)
        validation[corpus] = torch.load(
            val_path, map_location="cpu", weights_only=False
        )
        for role, path in (("train", train_path), ("validation", val_path)):
            sidecar = json.loads(Path(str(path) + ".json").read_text())
            pinned = source_by[corpus]
            expected_start = int(pinned[f"{role}_start"])
            expected_count = int(pinned[f"{role}_documents"])
            checks = {
                "split": role,
                "source_start": expected_start,
                "requested": expected_count,
                "corpus_id": corpus,
                "dataset": pinned["dataset_name"],
                "config": pinned.get("dataset_config"),
                "revision": pinned["dataset_revision"],
                "model": source["model_name"],
                "model_revision": source["model_revision"],
            }
            for key, expected in checks.items():
                if sidecar.get(key) != expected:
                    raise ValueError(
                        f"cache sidecar mismatch {path} {key}: {sidecar.get(key)!r} != {expected!r}"
                    )
            sidecars[f"{corpus}/{role}"] = {
                **sidecar,
                "sha256": sha256_file(Path(str(path) + ".json")),
            }
        for role, rows in (
            ("train", train[corpus]),
            ("validation", validation[corpus]),
        ):
            for row in rows:
                if not str(row["id"]).startswith(corpus + ":"):
                    raise ValueError(f"unprefixed row ID in {corpus}/{role}")
                if (
                    not torch.isfinite(row["leaves"]).all()
                    or not torch.isfinite(row["lengths"]).all()
                ):
                    raise FloatingPointError(
                        f"non-finite cache row {corpus}/{role}/{row['id']}"
                    )
                if row["leaves"].shape[0] < 2 or (row["lengths"] <= 0).any():
                    raise ValueError(f"invalid cache row {corpus}/{role}/{row['id']}")
    train_docs = {source_document_id(r["id"]) for rows in train.values() for r in rows}
    val_docs = {
        source_document_id(r["id"]) for rows in validation.values() for r in rows
    }
    overlap = train_docs & val_docs
    if overlap:
        raise AssertionError(
            f"training/validation document leakage: {sorted(overlap)[:3]}"
        )
    return train, validation, sidecars


def tree_plan(count: int, template: str = "balanced") -> list[list[int]]:
    if count < 2:
        return []
    if template == "balanced":
        plan = []
        while count > 1:
            positions = list(range(0, count - 1, 2))
            plan.append(positions)
            count = len(positions) + count % 2
        return plan
    plan = []
    while count > 1:
        if template == "left":
            position = 0
        elif template == "right":
            position = count - 2
        elif template == "outside_in":
            position = 0 if len(plan) % 2 == 0 else count - 2
        else:
            raise ValueError(f"unknown tree template: {template}")
        plan.append([position])
        count -= 1
    return plan


@dataclass(frozen=True)
class TopologyNode:
    leaf: int | None
    ordinal: int | None
    left: "TopologyNode | None"
    right: "TopologyNode | None"
    depth: int
    leaves: tuple[int, ...]


def topology_for_row(row: dict) -> tuple[TopologyNode, list[dict]]:
    nodes = [
        TopologyNode(i, None, None, None, 0, (i,))
        for i in range(row["leaves"].shape[0])
    ]
    sizes = [int(x) for x in row["lengths"].tolist()]
    records, ordinal = [], 0
    plan = tree_plan(len(nodes), row.get("tree_template", "balanced"))
    for positions in plan:
        made = []
        for pos in positions:
            left, right = nodes[pos], nodes[pos + 1]
            parent = TopologyNode(
                None,
                ordinal,
                left,
                right,
                max(left.depth, right.depth) + 1,
                left.leaves + right.leaves,
            )
            made.append((pos, parent, sizes[pos] + sizes[pos + 1]))
            records.append(
                {
                    "ordinal": ordinal,
                    "depth": parent.depth,
                    "left_leaves": left.leaves,
                    "right_leaves": right.leaves,
                    "parent_leaves": parent.leaves,
                    "left_is_leaf": left.leaf is not None,
                    "right_is_leaf": right.leaf is not None,
                }
            )
            ordinal += 1
        for pos, parent, size in reversed(made):
            nodes[pos : pos + 2] = [parent]
            sizes[pos : pos + 2] = [size]
    if len(nodes) != 1 or ordinal != row["leaves"].shape[0] - 1:
        raise AssertionError("tree did not reduce to one ordered root")
    flattened = []

    def visit(node):
        if node.leaf is not None:
            flattened.append(node.leaf)
        else:
            visit(node.left)
            visit(node.right)

    visit(nodes[0])
    if flattened != list(range(row["leaves"].shape[0])):
        raise AssertionError("recursive tree changed leaf ordering")
    return nodes[0], records


def validate_long_rows(rows: list[dict]) -> None:
    """Long retention rows may contain leaves/counts/provenance, never teacher roots."""
    forbidden = {
        "target",
        "targets",
        "teacher_root",
        "full_span_embedding",
        "text",
        "content",
    }
    for row in rows:
        overlap = forbidden & set(row)
        if overlap:
            raise ValueError(
                f"long retention row contains forbidden/fake teacher data: {sorted(overlap)}"
            )
        required = {
            "id",
            "corpus",
            "leaves",
            "lengths",
            "token_count",
            "source_start",
            "source_count",
        }
        if not required <= set(row):
            raise ValueError(
                f"long retention row lacks required provenance: {sorted(required - set(row))}"
            )
        if int(row["token_count"]) <= 6144:
            raise ValueError("long retention row is not longer than 6144 tokens")
        if row["leaves"].dtype != torch.float16:
            raise ValueError("long leaf embeddings must be fp16")
        if row["leaves"].shape[0] != row["lengths"].shape[0]:
            raise ValueError("long leaf/count mismatch")


def token_bucket(
    total: float, limits=(512, 1024, 2048, 4096, 8192, 16384, 32768)
) -> str:
    value = int(round(float(total)))
    previous = 0
    for limit in limits:
        if value <= limit:
            return f"<={limit}" if previous == 0 else f"{previous + 1}-{limit}"
        previous = limit
    return ">32768"


class CommonDecoderProbe(nn.Module):
    """Parent-only decoder: no length/weight/importance argument exists."""

    def __init__(self, dimension: int, hidden: int = 1024):
        super().__init__()
        self.dimension = int(dimension)
        self.net = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dimension),
        )

    def forward(self, parent: torch.Tensor) -> torch.Tensor:
        parent = F.normalize(parent, dim=-1)
        delta = self.net(parent).reshape(-1, 2, self.dimension)
        return F.normalize(
            parent[:, None, :] + 0.25 * torch.tanh(delta) / math.sqrt(self.dimension),
            dim=-1,
        )


class PairClassifierProbe(nn.Module):
    """Representation-pair classifier: metadata has no entry point."""

    def __init__(self, dimension: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4 * dimension),
            nn.Linear(4 * dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, root: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        root, candidate = F.normalize(root, dim=-1), F.normalize(candidate, dim=-1)
        x = torch.cat(
            (root, candidate, (root - candidate).abs(), root * candidate), dim=-1
        )
        return self.net(x).squeeze(-1)


def initialized_probe(
    kind: str, dimension: int, cfg: dict, device: torch.device
) -> tuple[nn.Module, str]:
    torch.manual_seed(int(cfg["probe_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["probe_seed"]))
    if kind == "common":
        model = CommonDecoderProbe(dimension, int(cfg["common_probe"]["hidden"]))
    else:
        model = PairClassifierProbe(dimension, int(cfg["classifier_probes"]["hidden"]))
    raw = b"".join(
        v.detach().cpu().contiguous().numpy().tobytes()
        for v in model.state_dict().values()
    )
    return model.to(device), hashlib.sha256(raw).hexdigest()


def build_merger(checkpoint_path: Path, device: torch.device) -> GeodesicMerge:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    autoencoder = checkpoint.get("variant") == "autoencoder"
    model = GeodesicMerge(
        d=next(
            v for k, v in checkpoint["state_dict"].items() if k == "input.0.weight"
        ).shape[1]
        - 2,
        hidden=int(cfg["encoder_hidden"]),
        layers=int(cfg["encoder_layers"]),
        heads=int(cfg["heads"]),
        max_angle=float(cfg.get("max_geodesic_angle", 0.8)),
        decoder=autoencoder,
        decoder_layers=int(cfg["decoder_layers"]),
        decoder_length_mode=str(cfg.get("decoder_length_mode", "oracle")),
        ratio_log_bound=float(cfg.get("ratio_log_bound", 4.0)),
        ratio_loss_weight=float(cfg.get("ratio_loss_weight", 0.0)),
        relational_loss_weight=float(cfg.get("relational_loss_weight", 0.0)),
        importance_mode=str(cfg.get("importance_mode", "token")),
        importance_beta=float(cfg.get("importance_beta", 1.0)),
        decoder_mode=str(cfg.get("decoder_mode", "legacy")),
        importance_loss_weight=float(cfg.get("importance_loss_weight", 0.0)),
        importance_prior_weight=float(cfg.get("importance_prior_weight", 0.0)),
        delta_loss_weight=float(cfg.get("delta_loss_weight", 0.0)),
        sharpness_loss_weight=float(cfg.get("sharpness_loss_weight", 0.0)),
        slot_loss_weight=float(cfg.get("slot_loss_weight", 0.0)),
        sharpness_margin=float(cfg.get("sharpness_margin", 0.1)),
        geodesic_parameterization=str(cfg.get("geodesic_parameterization", "legacy")),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def triple_universe(rows_by: dict[str, list]) -> dict[str, np.ndarray]:
    result = {}
    for ci, corpus in enumerate(CORPORA):
        refs = [
            (ci, ri, ordinal)
            for ri, row in enumerate(rows_by[corpus])
            for ordinal in range(row["leaves"].shape[0] - 1)
        ]
        result[corpus] = np.asarray(refs, dtype=np.int32)
    return result


def _sample_stratified_refs(
    rng: np.random.Generator,
    universe: dict[str, np.ndarray],
    updates: int,
    per_corpus: int,
) -> np.ndarray:
    chunks = []
    for _ in range(updates):
        batch = []
        for corpus in CORPORA:
            values = universe[corpus]
            chosen = rng.integers(0, len(values), size=per_corpus)
            batch.append(values[chosen])
        chunks.append(np.concatenate(batch, axis=0))
    return np.stack(chunks)


def _hard_negative_rows(
    rows: list,
    positives: np.ndarray,
    rng: np.random.Generator,
    candidate_count: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Select hardest of fixed random same-corpus candidates in leaf space."""
    n = len(positives)
    neg_rows = np.empty(n, dtype=np.int32)
    neg_leaves = np.empty(n, dtype=np.int16)
    for start in range(0, n, 512):
        block = positives[start : start + 512]
        candidates = rng.integers(
            0, len(rows), size=(len(block), candidate_count), dtype=np.int32
        )
        # Deterministically repair same-document candidates.
        for i, pos_row in enumerate(block):
            pos_doc = source_document_id(rows[int(pos_row)]["id"])
            for j in range(candidate_count):
                attempts = 0
                while source_document_id(rows[int(candidates[i, j])]["id"]) == pos_doc:
                    candidates[i, j] = (int(candidates[i, j]) + 1 + attempts) % len(
                        rows
                    )
                    attempts += 1
                    if attempts > len(rows):
                        raise RuntimeError("no disjoint negative document")
        candidate_leaf = np.zeros_like(candidates, dtype=np.int16)
        pos_vectors, candidate_vectors = [], []
        for i, pos_row in enumerate(block):
            prow = rows[int(pos_row)]
            pli = int(rng.integers(0, prow["leaves"].shape[0]))
            pos_vectors.append(prow["leaves"][pli].float())
            vectors = []
            for j, candidate_row in enumerate(candidates[i]):
                crow = rows[int(candidate_row)]
                li = int(rng.integers(0, crow["leaves"].shape[0]))
                candidate_leaf[i, j] = li
                vectors.append(crow["leaves"][li].float())
            candidate_vectors.append(torch.stack(vectors))
        p = F.normalize(torch.stack(pos_vectors), dim=-1)
        c = F.normalize(torch.stack(candidate_vectors), dim=-1)
        best = (c * p[:, None, :]).sum(-1).argmax(-1).numpy()
        index = np.arange(len(block))
        neg_rows[start : start + len(block)] = candidates[index, best]
        neg_leaves[start : start + len(block)] = candidate_leaf[index, best]
    return neg_rows, neg_leaves


def generate_schedules(train: dict, cfg: dict, smoke: bool = False) -> dict:
    rng = np.random.default_rng(int(cfg["seed"]))
    universe = triple_universe(train)
    common_updates = 1 if smoke else int(cfg["common_probe"]["updates"])
    classifier_updates = 1 if smoke else int(cfg["classifier_probes"]["updates"])
    per_corpus = 2 if smoke else int(cfg["common_probe"]["per_corpus_batch"])
    common = _sample_stratified_refs(rng, universe, common_updates, per_corpus)
    order_per_corpus = (
        2
        if smoke
        else int(cfg["classifier_probes"]["positive_pairs_per_update"]) // len(CORPORA)
    )
    order = _sample_stratified_refs(rng, universe, classifier_updates, order_per_corpus)

    (
        membership_rows,
        membership_pos_leaf,
        membership_neg_rows,
        membership_neg_leaf,
        membership_hard,
    ) = [], [], [], [], []
    positives_per_corpus = (
        2
        if smoke
        else int(cfg["classifier_probes"]["positive_pairs_per_update"]) // len(CORPORA)
    )
    # Generate by corpus in update order, then concatenate each update.
    corpus_parts = []
    for ci, corpus in enumerate(CORPORA):
        rows = train[corpus]
        positive = rng.integers(
            0, len(rows), size=classifier_updates * positives_per_corpus, dtype=np.int32
        )
        pos_leaf = np.asarray(
            [rng.integers(0, rows[int(r)]["leaves"].shape[0]) for r in positive],
            dtype=np.int16,
        )
        hard_mask = np.arange(len(positive)) % 2 == 1
        negative = rng.integers(0, len(rows), size=len(positive), dtype=np.int32)
        negative_leaf = np.asarray(
            [rng.integers(0, rows[int(r)]["leaves"].shape[0]) for r in negative],
            dtype=np.int16,
        )
        for i, pos in enumerate(positive):
            attempts = 0
            while source_document_id(
                rows[int(negative[i])]["id"]
            ) == source_document_id(rows[int(pos)]["id"]):
                negative[i] = (int(negative[i]) + 1 + attempts) % len(rows)
                attempts += 1
            # A repaired row can have a different leaf count from the original.
            negative_leaf[i] = int(
                rng.integers(0, rows[int(negative[i])]["leaves"].shape[0])
            )
        if hard_mask.any():
            hard_rows, hard_leaves = _hard_negative_rows(rows, positive[hard_mask], rng)
            negative[hard_mask], negative_leaf[hard_mask] = hard_rows, hard_leaves
        corpus_parts.append(
            (
                positive.reshape(classifier_updates, positives_per_corpus),
                pos_leaf.reshape(classifier_updates, positives_per_corpus),
                negative.reshape(classifier_updates, positives_per_corpus),
                negative_leaf.reshape(classifier_updates, positives_per_corpus),
                hard_mask.reshape(classifier_updates, positives_per_corpus),
                ci,
            )
        )
    for update in range(classifier_updates):
        membership_rows.append(
            np.concatenate(
                [
                    np.column_stack(
                        (np.full(positives_per_corpus, p[5], np.int32), p[0][update])
                    )
                    for p in corpus_parts
                ]
            )
        )
        membership_pos_leaf.append(np.concatenate([p[1][update] for p in corpus_parts]))
        membership_neg_rows.append(
            np.concatenate(
                [
                    np.column_stack(
                        (np.full(positives_per_corpus, p[5], np.int32), p[2][update])
                    )
                    for p in corpus_parts
                ]
            )
        )
        membership_neg_leaf.append(np.concatenate([p[3][update] for p in corpus_parts]))
        membership_hard.append(np.concatenate([p[4][update] for p in corpus_parts]))
    result = {
        "common": common.astype(np.int32),
        "order": order.astype(np.int32),
        "membership_positive_rows": np.stack(membership_rows).astype(np.int32),
        "membership_positive_leaf": np.stack(membership_pos_leaf).astype(np.int16),
        "membership_negative_rows": np.stack(membership_neg_rows).astype(np.int32),
        "membership_negative_leaf": np.stack(membership_neg_leaf).astype(np.int16),
        "membership_hard": np.stack(membership_hard).astype(bool),
    }
    return result


def schedule_manifest(schedules: dict) -> dict:
    result = {}
    for name, array in schedules.items():
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    result["combined_sha256"] = stable_hash(result)
    return result


@dataclass
class FeatureBundle:
    arm: str
    train_keys: np.ndarray
    train_triples: torch.Tensor
    train_root_keys: np.ndarray
    train_roots: torch.Tensor
    validation_triples: torch.Tensor
    validation_meta: list[dict]
    validation_roots: dict[tuple[int, int], torch.Tensor]
    validation_builtin: dict[str, torch.Tensor] | None
    controls: dict

    def train_triple_map(self) -> dict[tuple[int, int, int], int]:
        return {tuple(int(x) for x in key): i for i, key in enumerate(self.train_keys)}

    def train_root_map(self) -> dict[tuple[int, int], int]:
        return {
            tuple(int(x) for x in key): i for i, key in enumerate(self.train_root_keys)
        }


def selected_feature_keys(
    schedules: dict,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int]]]:
    triples = {tuple(int(x) for x in ref) for ref in schedules["common"].reshape(-1, 3)}
    triples.update(
        tuple(int(x) for x in ref) for ref in schedules["order"].reshape(-1, 3)
    )
    roots = {
        tuple(int(x) for x in ref)
        for ref in schedules["membership_positive_rows"].reshape(-1, 2)
    }
    return triples, roots


def _merge(
    model, arm: str, children: torch.Tensor, lengths: torch.Tensor, return_aux=False
):
    if arm == "pooling":
        value = weighted_average(children, lengths)
        if return_aux:
            mass = lengths / lengths.sum(1, keepdim=True)
            return value, {"effective_weights": mass}
        return value
    return model.merge(children, lengths, return_aux=return_aux)


@torch.inference_mode()
def extract_rows(
    rows: list,
    row_indices: list[int],
    arm: str,
    model,
    device: torch.device,
    needed: dict[int, set[int]] | None,
    need_roots: set[int],
    corpus_index: int,
    validation: bool = False,
    batch_size: int = 128,
):
    """Batch rows with equal topology; return packed selected triples and roots."""
    grouped = defaultdict(list)
    for ri in row_indices:
        row = rows[ri]
        grouped[(row["leaves"].shape[0], row.get("tree_template", "balanced"))].append(
            ri
        )
    triples, triple_keys, metas, roots, builtin = {}, [], [], {}, defaultdict(list)
    limits = (512, 1024, 2048, 4096, 8192, 16384, 32768)
    for (leaf_count, template), indices in sorted(grouped.items()):
        for start in range(0, len(indices), batch_size):
            part = indices[start : start + batch_size]
            nodes = torch.stack([rows[ri]["leaves"] for ri in part]).to(device).float()
            sizes = torch.stack([rows[ri]["lengths"] for ri in part]).to(device).float()
            descriptions = []
            for ri in part:
                _, rec = topology_for_row(rows[ri])
                descriptions.append(rec)
            ordinal = 0
            for positions in tree_plan(leaf_count, template):
                position = torch.tensor(positions, device=device)
                children = torch.stack(
                    (nodes[:, position], nodes[:, position + 1]), dim=2
                )
                child_sizes = torch.stack(
                    (sizes[:, position], sizes[:, position + 1]), dim=2
                )
                flat_children = children.reshape(-1, 2, nodes.shape[-1])
                flat_sizes = child_sizes.reshape(-1, 2)
                if arm == "autoencoder" and validation:
                    flat_parent, aux = _merge(
                        model, arm, flat_children, flat_sizes, return_aux=True
                    )
                else:
                    flat_parent = _merge(model, arm, flat_children, flat_sizes)
                    aux = None
                parent = flat_parent.reshape(len(part), len(positions), -1)
                for bi, ri in enumerate(part):
                    for pi, _ in enumerate(positions):
                        current = ordinal + pi
                        keep = validation or (
                            needed is not None and current in needed.get(ri, set())
                        )
                        if keep:
                            key = (corpus_index, ri, current)
                            triples[key] = (
                                torch.cat(
                                    (parent[bi, pi : pi + 1], children[bi, pi]), dim=0
                                )
                                .half()
                                .cpu()
                            )
                            triple_keys.append(key)
                            desc = descriptions[bi][current]
                            metas.append(
                                {
                                    "corpus": CORPORA[corpus_index],
                                    "corpus_index": corpus_index,
                                    "row_index": ri,
                                    "ordinal": current,
                                    "document": source_document_id(rows[ri]["id"]),
                                    "row_id": str(rows[ri]["id"]),
                                    "depth": int(desc["depth"]),
                                    "parent_token_bucket": token_bucket(
                                        float(child_sizes[bi, pi].sum()), limits
                                    ),
                                    "parent_tokens": int(
                                        round(float(child_sizes[bi, pi].sum()))
                                    ),
                                    **desc,
                                }
                            )
                            if arm == "autoencoder" and validation:
                                flat_index = bi * len(positions) + pi
                                decoded, daux = model.decode(
                                    parent[bi, pi : pi + 1], return_aux=True
                                )
                                predicted_mass = daux["importance_weights"]
                                cycled = model.merge(decoded, predicted_mass)
                                target_mass = aux["effective_weights"][flat_index]
                                builtin["decoded"].append(decoded[0].half().cpu())
                                builtin["predicted_mass"].append(
                                    predicted_mass[0].half().cpu()
                                )
                                builtin["target_mass"].append(target_mass.half().cpu())
                                builtin["cycle"].append(cycled[0].half().cpu())
                # Replace disjoint pairs while preserving carries.
                next_nodes, next_sizes, cursor = [], [], 0
                for pi, pos in enumerate(positions):
                    if cursor < pos:
                        next_nodes.append(nodes[:, cursor:pos])
                        next_sizes.append(sizes[:, cursor:pos])
                    next_nodes.append(parent[:, pi : pi + 1])
                    next_sizes.append(child_sizes[:, pi].sum(-1, keepdim=True))
                    cursor = pos + 2
                if cursor < nodes.shape[1]:
                    next_nodes.append(nodes[:, cursor:])
                    next_sizes.append(sizes[:, cursor:])
                nodes, sizes = torch.cat(next_nodes, 1), torch.cat(next_sizes, 1)
                ordinal += len(positions)
            for bi, ri in enumerate(part):
                if validation or ri in need_roots:
                    roots[(corpus_index, ri)] = nodes[bi, 0].half().cpu()
    ordered_keys = sorted(triples)
    packed = (
        torch.stack([triples[key] for key in ordered_keys])
        if ordered_keys
        else torch.empty(0, 3, 768, dtype=torch.float16)
    )
    meta_by = {(m["corpus_index"], m["row_index"], m["ordinal"]): m for m in metas}
    ordered_meta = [meta_by[key] for key in ordered_keys]
    packed_builtin = None
    if builtin:
        packed_builtin = {key: torch.stack(value) for key, value in builtin.items()}
        # Built-in values were appended traversal order, identical to metas before sorting.
        old_keys = triple_keys
        index = {key: i for i, key in enumerate(old_keys)}
        packed_builtin = {
            name: value[torch.tensor([index[k] for k in ordered_keys])]
            for name, value in packed_builtin.items()
        }
    return (
        np.asarray(ordered_keys, dtype=np.int32),
        packed,
        ordered_meta,
        roots,
        packed_builtin,
    )


def collapse_controls(roots: torch.Tensor) -> dict:
    x = roots.float()
    if not len(x):
        return {"count": 0}
    finite = torch.isfinite(x).all(-1)
    x = x[finite]
    norm = x.norm(dim=-1)
    sample = F.normalize(x[: min(2048, len(x))], dim=-1)
    centered = sample - sample.mean(0)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    effective_rank = float(
        (variance.sum().square() / variance.square().sum().clamp_min(1e-12)).item()
    )
    pair = sample[: min(512, len(sample))]
    sim = pair @ pair.T
    off = (
        sim[~torch.eye(len(pair), dtype=torch.bool)]
        if len(pair) > 1
        else torch.tensor([1.0])
    )
    unique = torch.unique((sample * 1e4).round().to(torch.int32), dim=0).shape[0]
    return {
        "count": int(len(roots)),
        "finite_fraction": float(finite.float().mean()),
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std(unbiased=False)),
        "norm_min": float(norm.min()),
        "norm_max": float(norm.max()),
        "mean_dimension_variance": float(centered.var(0, unbiased=False).mean()),
        "effective_rank": effective_rank,
        "mean_off_diagonal_cosine": float(off.mean()),
        "unique_rounded_fraction": float(unique / len(sample)),
    }


def build_feature_bundle(
    arm: str,
    train: dict,
    validation: dict,
    schedules: dict,
    cfg: dict,
    root: Path,
    device: torch.device,
) -> FeatureBundle:
    triple_set, root_set = selected_feature_keys(schedules)
    model = None
    if arm != "pooling":
        model = build_merger(root / cfg["frozen"][arm]["checkpoint"], device)
    train_key_parts, train_tensor_parts, train_roots = [], [], {}
    for ci, corpus in enumerate(CORPORA):
        needed = defaultdict(set)
        for c, ri, ordinal in triple_set:
            if c == ci:
                needed[ri].add(ordinal)
        root_rows = {ri for c, ri in root_set if c == ci}
        row_indices = sorted(set(needed) | root_rows)
        keys, packed, _, roots, _ = extract_rows(
            train[corpus],
            row_indices,
            arm,
            model,
            device,
            needed,
            root_rows,
            ci,
            validation=False,
        )
        train_key_parts.append(keys)
        train_tensor_parts.append(packed)
        train_roots.update(roots)
    train_keys = (
        np.concatenate(train_key_parts)
        if train_key_parts
        else np.empty((0, 3), np.int32)
    )
    train_triples = (
        torch.cat(train_tensor_parts)
        if train_tensor_parts
        else torch.empty(0, 3, 768, dtype=torch.float16)
    )
    order = np.lexsort((train_keys[:, 2], train_keys[:, 1], train_keys[:, 0]))
    train_keys, train_triples = (
        train_keys[order],
        train_triples[torch.from_numpy(order)],
    )
    root_keys = np.asarray(sorted(train_roots), dtype=np.int32)
    root_tensor = torch.stack([train_roots[tuple(key)] for key in root_keys])

    val_parts, val_meta, val_roots, builtin_parts = [], [], {}, defaultdict(list)
    for ci, corpus in enumerate(CORPORA):
        rows = validation[corpus]
        keys, packed, meta, roots, builtin = extract_rows(
            rows,
            list(range(len(rows))),
            arm,
            model,
            device,
            None,
            set(),
            ci,
            validation=True,
        )
        val_parts.append(packed)
        val_meta.extend(meta)
        val_roots.update(roots)
        if builtin:
            for key, value in builtin.items():
                builtin_parts[key].append(value)
    validation_triples = torch.cat(val_parts)
    validation_builtin = (
        {key: torch.cat(value) for key, value in builtin_parts.items()}
        if builtin_parts
        else None
    )
    root_control_tensor = torch.stack([val_roots[key] for key in sorted(val_roots)])
    controls = collapse_controls(root_control_tensor)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return FeatureBundle(
        arm,
        train_keys,
        train_triples,
        root_keys,
        root_tensor,
        validation_triples,
        val_meta,
        val_roots,
        validation_builtin,
        controls,
    )


def validation_membership_panels(
    validation: dict, seed: int, candidate_count: int = 64
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 117)
    outputs = {"random": [], "hard": []}
    for ci, corpus in enumerate(CORPORA):
        rows = validation[corpus]
        pos_leaf = np.asarray(
            [rng.integers(0, row["leaves"].shape[0]) for row in rows], dtype=np.int16
        )
        random_rows = rng.integers(0, len(rows), size=len(rows), dtype=np.int32)
        random_leaf = np.asarray(
            [rng.integers(0, rows[int(r)]["leaves"].shape[0]) for r in random_rows],
            dtype=np.int16,
        )
        for i in range(len(rows)):
            attempts = 0
            while source_document_id(
                rows[int(random_rows[i])]["id"]
            ) == source_document_id(rows[i]["id"]):
                random_rows[i] = (int(random_rows[i]) + 1 + attempts) % len(rows)
                attempts += 1
            random_leaf[i] = int(
                rng.integers(0, rows[int(random_rows[i])]["leaves"].shape[0])
            )
        hard_rows, hard_leaf = (
            np.empty(len(rows), np.int32),
            np.empty(len(rows), np.int16),
        )
        for start in range(0, len(rows), 256):
            stop = min(len(rows), start + 256)
            pvec, cvec, crows, cleaves = [], [], [], []
            for i in range(start, stop):
                pvec.append(rows[i]["leaves"][int(pos_leaf[i])].float())
                these_rows, these_leaves, vectors = [], [], []
                while len(these_rows) < candidate_count:
                    cr = int(rng.integers(0, len(rows)))
                    if source_document_id(rows[cr]["id"]) == source_document_id(
                        rows[i]["id"]
                    ):
                        continue
                    cl = int(rng.integers(0, rows[cr]["leaves"].shape[0]))
                    these_rows.append(cr)
                    these_leaves.append(cl)
                    vectors.append(rows[cr]["leaves"][cl].float())
                crows.append(these_rows)
                cleaves.append(these_leaves)
                cvec.append(torch.stack(vectors))
            p = F.normalize(torch.stack(pvec), dim=-1)
            c = F.normalize(torch.stack(cvec), dim=-1)
            best = (p[:, None] * c).sum(-1).argmax(-1).tolist()
            for local, selected in enumerate(best):
                hard_rows[start + local] = crows[local][selected]
                hard_leaf[start + local] = cleaves[local][selected]
        for i in range(len(rows)):
            outputs["random"].append(
                (ci, i, int(pos_leaf[i]), int(random_rows[i]), int(random_leaf[i]))
            )
            outputs["hard"].append(
                (ci, i, int(pos_leaf[i]), int(hard_rows[i]), int(hard_leaf[i]))
            )
    return {key: np.asarray(value, dtype=np.int32) for key, value in outputs.items()}


def validation_leaf_index(
    validation: dict,
) -> tuple[np.ndarray, list[dict], torch.Tensor]:
    refs, meta, vectors = [], [], []
    for ci, corpus in enumerate(CORPORA):
        for ri, row in enumerate(validation[corpus]):
            root_bucket = token_bucket(float(row["lengths"].sum()))
            topology, _ = topology_for_row(row)
            depths = {}

            def walk(node, depth):
                if node.leaf is not None:
                    depths[node.leaf] = depth
                else:
                    walk(node.left, depth + 1)
                    walk(node.right, depth + 1)

            walk(topology, 0)
            for li, vector in enumerate(row["leaves"]):
                refs.append((ci, ri, li))
                vectors.append(vector.half())
                meta.append(
                    {
                        "corpus": corpus,
                        "corpus_index": ci,
                        "row_index": ri,
                        "leaf_index": li,
                        "document": source_document_id(row["id"]),
                        "row_id": str(row["id"]),
                        "depth": int(depths[li]),
                        "parent_token_bucket": root_bucket,
                    }
                )
    return np.asarray(refs, np.int32), meta, torch.stack(vectors)


@torch.inference_mode()
def make_hard_panels(
    targets: torch.Tensor,
    meta: list[dict],
    panel_size: int,
    neighbors_considered: int,
    device: torch.device,
) -> np.ndarray:
    """Fixed pooling-space panels with deterministic staged stratum relaxation."""
    n = len(meta)
    panel = np.full((n, panel_size), -1, dtype=np.int32)
    panel[:, 0] = np.arange(n, dtype=np.int32)
    normalized = F.normalize(targets.float(), dim=-1)
    unresolved = {i: panel_size - 1 for i in range(n)}
    # Strict corpus/depth/bucket, then corpus/depth, then corpus.
    key_functions = (
        lambda m: (m["corpus"], m["depth"], m["parent_token_bucket"]),
        lambda m: (m["corpus"], m["depth"]),
        lambda m: (m["corpus"],),
    )
    for key_fn in key_functions:
        groups = defaultdict(list)
        for i, item in enumerate(meta):
            groups[key_fn(item)].append(i)
        for indices in groups.values():
            queries = [i for i in indices if unresolved.get(i, 0) > 0]
            if not queries or len(indices) < 2:
                continue
            candidates = torch.tensor(indices, dtype=torch.long)
            c = normalized[candidates].to(device)
            for start in range(0, len(queries), 256):
                qindices = queries[start : start + 256]
                q = normalized[torch.tensor(qindices)].to(device)
                similarity = q @ c.T
                k = min(max(neighbors_considered, panel_size * 2), len(indices))
                top = (
                    similarity.topk(k, dim=1, largest=True, sorted=True)
                    .indices.cpu()
                    .numpy()
                )
                for local, qi in enumerate(qindices):
                    already = set(int(x) for x in panel[qi] if x >= 0)
                    write = int((panel[qi] >= 0).sum())
                    for position in top[local]:
                        candidate = indices[int(position)]
                        if (
                            candidate in already
                            or meta[candidate]["document"] == meta[qi]["document"]
                        ):
                            continue
                        panel[qi, write] = candidate
                        already.add(candidate)
                        write += 1
                        if write == panel_size:
                            break
                    unresolved[qi] = panel_size - write
            del c
    return panel


def save_panel_artifacts(
    out: Path,
    schedules: dict,
    membership_panels: dict,
    child_panel: np.ndarray,
    leaf_panel: np.ndarray,
) -> dict:
    schedule_path = out / "schedules.npz"
    np.savez_compressed(schedule_path, **schedules)
    panel_path = out / "validation_panels.npz"
    np.savez_compressed(
        panel_path,
        membership_random=membership_panels["random"],
        membership_hard=membership_panels["hard"],
        child_retrieval=child_panel,
        leaf_retrieval=leaf_panel,
    )
    return {
        "schedule_file": str(schedule_path.name),
        "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_arrays": schedule_manifest(schedules),
        "panel_file": str(panel_path.name),
        "panel_file_sha256": sha256_file(panel_path),
        "panel_arrays": {
            name: {
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name, array in (
                ("membership_random", membership_panels["random"]),
                ("membership_hard", membership_panels["hard"]),
                ("child_retrieval", child_panel),
                ("leaf_retrieval", leaf_panel),
            )
        },
    }


def _common_indices(bundle: FeatureBundle, refs: np.ndarray) -> torch.Tensor:
    mapping = bundle.train_triple_map()
    try:
        return torch.tensor(
            [mapping[tuple(int(x) for x in ref)] for ref in refs.reshape(-1, 3)],
            dtype=torch.long,
        ).reshape(refs.shape[:-1])
    except KeyError as exc:
        raise AssertionError(f"missing matched common/order feature: {exc}") from exc


def train_common(
    bundle: FeatureBundle,
    schedules: dict,
    cfg: dict,
    device: torch.device,
    out: Path,
    smoke: bool = False,
) -> tuple[CommonDecoderProbe, dict]:
    model, init_hash = initialized_probe(
        "common", bundle.train_triples.shape[-1], cfg, device
    )
    section = cfg["common_probe"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    indices = _common_indices(bundle, schedules["common"])
    writer = (
        SummaryWriter(out / "tensorboard" / f"common_{bundle.arm}")
        if SummaryWriter
        else None
    )
    losses = []
    model.train()
    for step in range(indices.shape[0]):
        batch = bundle.train_triples[indices[step]].to(device).float()
        parent, target = batch[:, 0], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(parent)
        cosine = 1 - F.cosine_similarity(prediction, target, dim=-1).mean()
        mse = F.mse_loss(prediction, target)
        loss = cosine + float(section["mse_weight"]) * mse
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(section["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
        if writer:
            writer.add_scalar("train/loss", losses[-1], step + 1)
    if writer:
        writer.close()
    checkpoint = out / "probes" / f"common_{bundle.arm}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arm": bundle.arm,
            "kind": "common",
            "init_sha256": init_hash,
            "updates": int(indices.shape[0]),
            "configured_updates": int(section["updates"]),
            "schedule_sha256": hashlib.sha256(
                schedules["common"].tobytes()
            ).hexdigest(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        checkpoint,
    )
    return model.eval(), {
        "arm": bundle.arm,
        "kind": "common",
        "init_sha256": init_hash,
        "updates": int(indices.shape[0]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "final_loss": losses[-1],
        "mean_loss": float(np.mean(losses)),
        "checkpoint": str(checkpoint.relative_to(out)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def train_classifier(
    kind: str,
    bundle: FeatureBundle,
    schedules: dict,
    train: dict,
    cfg: dict,
    device: torch.device,
    out: Path,
) -> tuple[PairClassifierProbe, dict]:
    model, init_hash = initialized_probe(
        kind, bundle.train_triples.shape[-1], cfg, device
    )
    section = cfg["classifier_probes"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    root_map, triple_map = bundle.train_root_map(), bundle.train_triple_map()
    writer = (
        SummaryWriter(out / "tensorboard" / f"{kind}_{bundle.arm}")
        if SummaryWriter
        else None
    )
    losses = []
    model.train()
    if kind == "membership":
        updates = schedules["membership_positive_rows"].shape[0]
    elif kind == "order":
        updates = schedules["order"].shape[0]
    else:
        raise ValueError(kind)
    for step in range(updates):
        if kind == "membership":
            positive_refs = schedules["membership_positive_rows"][step]
            negative_refs = schedules["membership_negative_rows"][step]
            roots, positives, negatives = [], [], []
            for j, (ci, ri) in enumerate(positive_refs):
                roots.append(bundle.train_roots[root_map[(int(ci), int(ri))]])
                positives.append(
                    train[CORPORA[int(ci)]][int(ri)]["leaves"][
                        int(schedules["membership_positive_leaf"][step, j])
                    ]
                )
                nci, nri = negative_refs[j]
                negatives.append(
                    train[CORPORA[int(nci)]][int(nri)]["leaves"][
                        int(schedules["membership_negative_leaf"][step, j])
                    ]
                )
            roots = torch.stack(roots).float()
            root_batch = torch.cat((roots, roots), 0).to(device)
            candidate = (
                torch.cat((torch.stack(positives), torch.stack(negatives)), 0)
                .float()
                .to(device)
            )
            label = torch.cat((torch.ones(len(roots)), torch.zeros(len(roots)))).to(
                device
            )
        else:
            refs = schedules["order"][step]
            features = torch.stack(
                [
                    bundle.train_triples[triple_map[tuple(int(x) for x in ref)]]
                    for ref in refs
                ]
            ).float()
            root_batch = torch.cat((features[:, 0], features[:, 0]), 0).to(device)
            candidate = torch.cat((features[:, 1], features[:, 2]), 0).to(device)
            label = torch.cat(
                (torch.zeros(len(features)), torch.ones(len(features)))
            ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(root_batch, candidate)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(section["gradient_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
        if writer:
            writer.add_scalar("train/loss", losses[-1], step + 1)
    if writer:
        writer.close()
    checkpoint = out / "probes" / f"{kind}_{bundle.arm}.pt"
    schedule_name = "order" if kind == "order" else "membership_positive_rows"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arm": bundle.arm,
            "kind": kind,
            "init_sha256": init_hash,
            "updates": updates,
            "schedule_sha256": hashlib.sha256(
                schedules[schedule_name].tobytes()
            ).hexdigest(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        checkpoint,
    )
    return model.eval(), {
        "arm": bundle.arm,
        "kind": kind,
        "init_sha256": init_hash,
        "updates": updates,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "final_loss": losses[-1],
        "mean_loss": float(np.mean(losses)),
        "checkpoint": str(checkpoint.relative_to(out)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _bootstrap_mean(
    values: np.ndarray, documents: np.ndarray, resamples: int, seed: int
) -> tuple[float, float]:
    unique, inverse = np.unique(documents, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    if len(unique) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, np.float64)
    for i in range(resamples):
        sample = rng.integers(0, len(unique), size=len(unique))
        estimates[i] = sums[sample].sum() / counts[sample].sum()
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def _bootstrap_classification(
    y: np.ndarray,
    score: np.ndarray,
    documents: np.ndarray,
    metric: str,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    unique = np.unique(documents)
    grouped = {doc: np.flatnonzero(documents == doc) for doc in unique}
    if len(unique) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([grouped[d] for d in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        if metric == "roc_auc":
            value = roc_auc_score(y[idx], score[idx])
        elif metric == "pr_auc":
            value = average_precision_score(y[idx], score[idx])
        else:
            value = balanced_accuracy_score(y[idx], score[idx] >= 0.5)
        estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def grouped_mean_summaries(
    rows: list[dict], arm: str, task: str, metric_names: Iterable[str], cfg: dict
) -> list[dict]:
    output = []
    slices = [("overall", "all", np.ones(len(rows), dtype=bool))]
    for field, slice_type in (
        ("corpus", "corpus"),
        ("depth", "depth"),
        ("parent_token_bucket", "parent_token_bucket"),
    ):
        values = sorted({str(row[field]) for row in rows})
        slices.extend(
            (slice_type, value, np.asarray([str(row[field]) == value for row in rows]))
            for value in values
        )
    documents = np.asarray([row["document"] for row in rows])
    for slice_type, name, mask in slices:
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows])[mask]
            docs = documents[mask]
            lo, hi = _bootstrap_mean(
                values,
                docs,
                int(cfg["bootstrap_resamples"]),
                int(cfg["bootstrap_seed"]) + len(output),
            )
            output.append(
                {
                    "arm": arm,
                    "task": task,
                    "slice_type": slice_type,
                    "slice": name,
                    "metric": metric,
                    "value": float(values.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "examples": int(len(values)),
                    "documents": int(len(np.unique(docs))),
                }
            )
    return output


def classification_summaries(
    rows: list[dict], arm: str, task: str, cfg: dict
) -> list[dict]:
    output = []
    slices = [("overall", "all", np.ones(len(rows), bool))]
    for field, slice_type in (
        ("corpus", "corpus"),
        ("parent_token_bucket", "parent_token_bucket"),
    ):
        for value in sorted({str(row[field]) for row in rows}):
            slices.append(
                (
                    slice_type,
                    value,
                    np.asarray([str(row[field]) == value for row in rows]),
                )
            )
    y = np.asarray([row["label"] for row in rows], float)
    score = np.asarray([row["score"] for row in rows], float)
    docs = np.asarray([row["document"] for row in rows])
    for slice_type, name, mask in slices:
        if len(np.unique(y[mask])) < 2:
            continue
        values = {
            "roc_auc": roc_auc_score(y[mask], score[mask]),
            "pr_auc": average_precision_score(y[mask], score[mask]),
            "balanced_accuracy": balanced_accuracy_score(y[mask], score[mask] >= 0.5),
        }
        for metric, value in values.items():
            lo, hi = _bootstrap_classification(
                y[mask],
                score[mask],
                docs[mask],
                metric,
                int(cfg["bootstrap_resamples"]),
                int(cfg["bootstrap_seed"]) + len(output),
            )
            output.append(
                {
                    "arm": arm,
                    "task": task,
                    "slice_type": slice_type,
                    "slice": name,
                    "metric": metric,
                    "value": float(value),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "examples": int(mask.sum()),
                    "documents": int(len(np.unique(docs[mask]))),
                }
            )
    return output


def retrieval_ranks(
    prediction: torch.Tensor,
    candidates: torch.Tensor,
    panel: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    prediction = F.normalize(prediction.float(), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    ranks = np.empty(len(panel), np.int32)
    for start in range(0, len(panel), 512):
        selected = panel[start : start + 512]
        valid = selected >= 0
        safe = np.maximum(selected, 0)
        target = candidates[torch.from_numpy(safe)].to(device)
        pred = prediction[start : start + len(selected)].to(device)
        score = (pred[:, None] * target).sum(-1).cpu().numpy()
        score[~valid] = -np.inf
        correct = score[:, 0]
        ids = selected
        greater = (score[:, 1:] > correct[:, None]).sum(1)
        tied_before = (
            (score[:, 1:] == correct[:, None])
            & (ids[:, 1:] < ids[:, :1])
            & valid[:, 1:]
        ).sum(1)
        ranks[start : start + len(selected)] = 1 + greater + tied_before
    return ranks


@torch.inference_mode()
def evaluate_common(
    model: CommonDecoderProbe,
    bundle: FeatureBundle,
    child_panel: np.ndarray,
    device: torch.device,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    feature = bundle.validation_triples.float()
    predictions = []
    for start in range(0, len(feature), 512):
        predictions.append(model(feature[start : start + 512, 0].to(device)).cpu())
    prediction = torch.cat(predictions)
    target = feature[:, 1:]
    flat_prediction, flat_target = (
        prediction.reshape(-1, prediction.shape[-1]),
        target.reshape(-1, target.shape[-1]),
    )
    ranks = retrieval_ranks(flat_prediction, flat_target, child_panel, device)
    cosine = F.cosine_similarity(flat_prediction, flat_target, dim=-1).clamp(-1, 1)
    mse = (flat_prediction - flat_target).square().mean(-1)
    angle = torch.rad2deg(torch.acos(cosine))
    rows = []
    for i in range(len(flat_prediction)):
        meta = dict(bundle.validation_meta[i // 2])
        meta["slot"] = i % 2
        rank = int(ranks[i])
        rows.append(
            {
                **meta,
                "cosine": float(cosine[i]),
                "mse": float(mse[i]),
                "angular_error_degrees": float(angle[i]),
                "recall_at_1": float(rank <= 1),
                "recall_at_5": float(rank <= 5),
                "recall_at_10": float(rank <= 10),
                "mrr": 1.0 / rank,
            }
        )
    summary = grouped_mean_summaries(
        rows,
        bundle.arm,
        "common_child",
        (
            "cosine",
            "mse",
            "angular_error_degrees",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr",
        ),
        cfg,
    )
    return rows, summary


@torch.inference_mode()
def evaluate_recursive(
    model: CommonDecoderProbe,
    bundle: FeatureBundle,
    validation: dict,
    leaf_meta: list[dict],
    leaf_vectors: torch.Tensor,
    leaf_panel: np.ndarray,
    device: torch.device,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    triple_index = {
        (m["corpus_index"], m["row_index"], m["ordinal"]): i
        for i, m in enumerate(bundle.validation_meta)
    }
    leaf_lookup = {
        (m["corpus_index"], m["row_index"], m["leaf_index"]): i
        for i, m in enumerate(leaf_meta)
    }
    predictions = torch.empty_like(leaf_vectors.float())
    direct = torch.empty(len(leaf_meta), dtype=torch.float32)
    rows = [None] * len(leaf_meta)
    for ci, corpus in enumerate(CORPORA):
        for ri, row in enumerate(validation[corpus]):
            topology, _ = topology_for_row(row)
            true_bottom_predictions = {}

            def collect_direct(node):
                if node.leaf is not None:
                    return
                ti = triple_index[(ci, ri, int(node.ordinal))]
                children = model(bundle.validation_triples[ti, 0:1].float().to(device))[
                    0
                ].cpu()
                if node.left.leaf is not None:
                    true_bottom_predictions[node.left.leaf] = children[0]
                if node.right.leaf is not None:
                    true_bottom_predictions[node.right.leaf] = children[1]
                collect_direct(node.left)
                collect_direct(node.right)

            collect_direct(topology)
            produced = {}

            def recurse(node, vector, depth):
                if node.leaf is not None:
                    produced[node.leaf] = (vector.cpu(), depth)
                    return
                children = model(vector[None].to(device).float())[0]
                recurse(node.left, children[0], depth + 1)
                recurse(node.right, children[1], depth + 1)

            recurse(topology, bundle.validation_roots[(ci, ri)], 0)
            for li in range(row["leaves"].shape[0]):
                global_index = leaf_lookup[(ci, ri, li)]
                prediction, depth = produced[li]
                target = row["leaves"][li].float()
                predictions[global_index] = prediction
                direct_cos = F.cosine_similarity(
                    true_bottom_predictions[li], target, dim=0
                )
                direct[global_index] = direct_cos
                cosine = F.cosine_similarity(prediction, target, dim=0).clamp(-1, 1)
                rows[global_index] = {
                    **leaf_meta[global_index],
                    "depth": depth,
                    "cosine": float(cosine),
                    "mse": float((prediction - target).square().mean()),
                    "angular_error_degrees": float(torch.rad2deg(torch.acos(cosine))),
                    "direct_bottom_cosine": float(direct_cos),
                    "cosine_degradation": float(cosine - direct_cos),
                }
    ranks = retrieval_ranks(predictions, leaf_vectors, leaf_panel, device)
    for i, rank in enumerate(ranks):
        rows[i].update(
            {
                "recall_at_1": float(rank <= 1),
                "recall_at_5": float(rank <= 5),
                "recall_at_10": float(rank <= 10),
                "mrr": 1.0 / int(rank),
            }
        )
    summary = grouped_mean_summaries(
        rows,
        bundle.arm,
        "recursive_leaf",
        (
            "cosine",
            "mse",
            "angular_error_degrees",
            "direct_bottom_cosine",
            "cosine_degradation",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "mrr",
        ),
        cfg,
    )
    return rows, summary


@torch.inference_mode()
def evaluate_membership(
    model: PairClassifierProbe,
    bundle: FeatureBundle,
    validation: dict,
    panels: dict[str, np.ndarray],
    device: torch.device,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for panel_name, panel in panels.items():
        roots, positive, negative, metadata = [], [], [], []
        for ci, ri, pli, nri, nli in panel:
            corpus = CORPORA[int(ci)]
            row = validation[corpus][int(ri)]
            neg = validation[corpus][int(nri)]
            roots.append(bundle.validation_roots[(int(ci), int(ri))])
            positive.append(row["leaves"][int(pli)])
            negative.append(neg["leaves"][int(nli)])
            metadata.append(
                {
                    "corpus": corpus,
                    "document": source_document_id(row["id"]),
                    "row_id": str(row["id"]),
                    "parent_token_bucket": token_bucket(float(row["lengths"].sum())),
                    "negative_panel": panel_name,
                }
            )
        roots = torch.stack(roots).float()
        candidates = torch.cat((torch.stack(positive), torch.stack(negative))).float()
        roots2 = torch.cat((roots, roots))
        scores = []
        for start in range(0, len(roots2), 512):
            scores.append(
                torch.sigmoid(
                    model(
                        roots2[start : start + 512].to(device),
                        candidates[start : start + 512].to(device),
                    )
                ).cpu()
            )
        scores = torch.cat(scores).numpy()
        labels = np.concatenate((np.ones(len(roots)), np.zeros(len(roots))))
        task_rows = []
        for i in range(len(labels)):
            meta = metadata[i % len(metadata)]
            task_rows.append(
                {**meta, "label": float(labels[i]), "score": float(scores[i])}
            )
        rows.extend(task_rows)
        summaries.extend(
            classification_summaries(
                task_rows, bundle.arm, f"membership_{panel_name}", cfg
            )
        )
    return rows, summaries


@torch.inference_mode()
def evaluate_order(
    model: PairClassifierProbe, bundle: FeatureBundle, device: torch.device, cfg: dict
) -> tuple[list[dict], list[dict]]:
    features = bundle.validation_triples.float()
    roots = torch.cat((features[:, 0], features[:, 0]))
    candidates = torch.cat((features[:, 1], features[:, 2]))
    labels = np.concatenate((np.zeros(len(features)), np.ones(len(features))))
    scores = []
    for start in range(0, len(roots), 512):
        scores.append(
            torch.sigmoid(
                model(
                    roots[start : start + 512].to(device),
                    candidates[start : start + 512].to(device),
                )
            ).cpu()
        )
    scores = torch.cat(scores).numpy()
    rows = []
    for i in range(len(labels)):
        meta = dict(bundle.validation_meta[i % len(features)])
        rows.append({**meta, "label": float(labels[i]), "score": float(scores[i])})
    summaries = classification_summaries(rows, bundle.arm, "pair_order", cfg)
    return rows, summaries


def evaluate_builtin_autoencoder(
    bundle: FeatureBundle, cfg: dict
) -> tuple[list[dict], list[dict]]:
    if bundle.arm != "autoencoder" or bundle.validation_builtin is None:
        return [], []
    true = bundle.validation_triples[:, 1:].float()
    decoded = bundle.validation_builtin["decoded"].float()
    parent = bundle.validation_triples[:, 0].float()
    cycled = bundle.validation_builtin["cycle"].float()
    predicted_mass = bundle.validation_builtin["predicted_mass"].float().clamp_min(1e-8)
    target_mass = bundle.validation_builtin["target_mass"].float().clamp_min(1e-8)
    child_cos = F.cosine_similarity(decoded, true, dim=-1).mean(-1).clamp(-1, 1)
    child_mse = (decoded - true).square().mean((1, 2))
    child_angle = torch.rad2deg(
        torch.acos(F.cosine_similarity(decoded, true, dim=-1).clamp(-1, 1))
    ).mean(-1)
    cycle_cos = F.cosine_similarity(cycled, parent, dim=-1)
    importance_kl = (target_mass * (target_mass.log() - predicted_mass.log())).sum(-1)
    entropy = -(predicted_mass * predicted_mass.log()).sum(-1)
    rows = []
    for i, meta in enumerate(bundle.validation_meta):
        rows.append(
            {
                **meta,
                "reconstruction_cosine": float(child_cos[i]),
                "reconstruction_mse": float(child_mse[i]),
                "reconstruction_angular_error_degrees": float(child_angle[i]),
                "cycle_cosine": float(cycle_cos[i]),
                "importance_kl": float(importance_kl[i]),
                "predicted_importance_entropy": float(entropy[i]),
            }
        )
    summary = grouped_mean_summaries(
        rows,
        bundle.arm,
        "builtin_autoencoder_capability",
        (
            "reconstruction_cosine",
            "reconstruction_mse",
            "reconstruction_angular_error_degrees",
            "cycle_cosine",
            "importance_kl",
            "predicted_importance_entropy",
        ),
        cfg,
    )
    return rows, summary


def metric_sanity_checks() -> dict:
    target = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    perfect = F.cosine_similarity(target, target).mean().item()
    swapped = F.cosine_similarity(target.flip(0), target).mean().item()
    y = np.array([0, 0, 1, 1])
    score = np.array([0.0, 0.1, 0.9, 1.0])
    values = {
        "perfect_cosine": perfect,
        "swapped_cosine": swapped,
        "perfect_roc_auc": float(roc_auc_score(y, score)),
        "perfect_pr_auc": float(average_precision_score(y, score)),
    }
    if values != {
        "perfect_cosine": 1.0,
        "swapped_cosine": 0.0,
        "perfect_roc_auc": 1.0,
        "perfect_pr_auc": 1.0,
    }:
        raise AssertionError(f"metric sanity failed: {values}")
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_value(
    summaries: list[dict], arm: str, task: str, metric: str
) -> dict | None:
    return next(
        (
            x
            for x in summaries
            if x["arm"] == arm
            and x["task"] == task
            and x["metric"] == metric
            and x["slice_type"] == "overall"
        ),
        None,
    )


def build_report(
    summaries: list[dict], controls: dict, run: dict, long_status: dict
) -> str:
    lines = [
        "# Validation-only information-retention report",
        "",
        "This study used only pinned multicorpus training and held-out validation roles. "
        "No frozen official test artifact/range and no model-attack test evaluation was accessed.",
        "",
        "## Fair common parent-only probe",
        "",
        "All arms used the same parent-only architecture, tensor-identical initialization, update budget, "
        "and sampling schedule. No child lengths, weights, importance, corpus, depth, or tree metadata entered a probe.",
        "",
        "| arm | child cosine (95% doc CI) | child R@1 | child MRR | recursive leaf cosine | recursive leaf R@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        vals = [
            summary_value(summaries, arm, "common_child", "cosine"),
            summary_value(summaries, arm, "common_child", "recall_at_1"),
            summary_value(summaries, arm, "common_child", "mrr"),
            summary_value(summaries, arm, "recursive_leaf", "cosine"),
            summary_value(summaries, arm, "recursive_leaf", "recall_at_1"),
        ]

        def fmt(x, ci=False):
            if not x:
                return "pending"
            return (
                f"{x['value']:.4f} [{x['ci95_low']:.4f}, {x['ci95_high']:.4f}]"
                if ci
                else f"{x['value']:.4f}"
            )

        lines.append(
            f"| {arm} | {fmt(vals[0], True)} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(vals[3], True)} | {fmt(vals[4])} |"
        )
    lines += [
        "",
        "## Additional fair probes",
        "",
        "| arm | membership random ROC/PR | membership hard ROC/PR | pair-order ROC/PR |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:

        def pair(task):
            a, p = (
                summary_value(summaries, arm, task, "roc_auc"),
                summary_value(summaries, arm, task, "pr_auc"),
            )
            return "pending" if not a else f"{a['value']:.4f} / {p['value']:.4f}"

        lines.append(
            f"| {arm} | {pair('membership_random')} | {pair('membership_hard')} | {pair('pair_order')} |"
        )
    builtin_recon = summary_value(
        summaries,
        "autoencoder",
        "builtin_autoencoder_capability",
        "reconstruction_cosine",
    )
    builtin_cycle = summary_value(
        summaries, "autoencoder", "builtin_autoencoder_capability", "cycle_cosine"
    )
    lines += [
        "",
        "## Built-in staged-AE decoder (system capability; not a fair-probe arm)",
        "",
        f"Built-in ordered-child reconstruction cosine: {builtin_recon['value']:.4f}."
        if builtin_recon
        else "Built-in reconstruction pending.",
        f"Built-in cycle cosine: {builtin_cycle['value']:.6f}."
        if builtin_cycle
        else "Built-in cycle pending.",
        "Cycle consistency alone is not evidence of information retention: a decoder and encoder can form a highly consistent cycle without recovering the original ordered children.",
        "",
        "## Collapse/norm controls",
        "",
        "```json",
        json.dumps(controls, indent=2, sort_keys=True),
        "```",
        "",
        "## Long validation",
        "",
        json.dumps(long_status, indent=2, sort_keys=True),
        "",
        "## Reproducibility",
        "",
        f"Run status: `{run.get('status')}`. Frozen hashes, schedules, panels, probe budgets, JSON/CSV, TensorBoard, and final file hashes are stored beside this report.",
        "",
        "For roots above 6,144 tokens the protocol permits only original short leaf embeddings as targets; it never fabricates a full-span teacher target.",
    ]
    return "\n".join(lines) + "\n"


def run_study(
    root: Path,
    config_path: Path,
    device: torch.device,
    smoke: bool = False,
    skip_long: bool = False,
) -> Path:
    started = time.time()
    cfg, source, verified = load_protocol(root, config_path)
    out = root / (cfg["output_dir"] + ("_smoke" if smoke else ""))
    if out.exists():
        if smoke:
            shutil.rmtree(out)
        else:
            raise FileExistsError(
                f"refusing to overwrite existing retention output: {out}"
            )
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    shutil.copy2(config_path, out / "config.yaml")
    shutil.copy2(
        root / "INFORMATION_RETENTION_PLAN.md", out / "INFORMATION_RETENTION_PLAN.md"
    )
    train, validation, sidecars = load_caches(root, cfg, source)
    if smoke:
        # Smoke artifacts are never study candidates; keep the CUDA path tiny.
        validation = {corpus: rows[:2] for corpus, rows in validation.items()}
        cfg = dict(cfg)
        cfg["bootstrap_resamples"] = 10
    sanity = metric_sanity_checks()
    schedules = generate_schedules(train, cfg, smoke=smoke)

    # Pooling validation creates arm-independent fixed hard-negative panels.
    bundles = {}
    for arm in ARMS:
        print(
            f"[{time.strftime('%H:%M:%S')}] extracting frozen features: {arm}",
            flush=True,
        )
        bundles[arm] = build_feature_bundle(
            arm, train, validation, schedules, cfg, root, device
        )
    pooling = bundles["pooling"]
    child_targets = pooling.validation_triples[:, 1:].reshape(
        -1, pooling.validation_triples.shape[-1]
    )
    child_meta = [
        dict(meta, slot=slot) for meta in pooling.validation_meta for slot in (0, 1)
    ]
    child_panel = make_hard_panels(
        child_targets,
        child_meta,
        int(cfg["retrieval"]["panel_size"]),
        int(cfg["retrieval"]["neighbors_considered"]),
        device,
    )
    leaf_refs, leaf_meta, leaf_vectors = validation_leaf_index(validation)
    leaf_panel = make_hard_panels(
        leaf_vectors,
        leaf_meta,
        int(cfg["retrieval"]["panel_size"]),
        int(cfg["retrieval"]["neighbors_considered"]),
        device,
    )
    membership_panels = validation_membership_panels(validation, int(cfg["seed"]))
    panel_manifest = save_panel_artifacts(
        out, schedules, membership_panels, child_panel, leaf_panel
    )

    all_rows, summaries, probe_runs = [], [], []
    common_models = {}
    common_init = []
    for arm in ARMS:
        print(
            f"[{time.strftime('%H:%M:%S')}] train/evaluate common probe: {arm}",
            flush=True,
        )
        model, record = train_common(bundles[arm], schedules, cfg, device, out, smoke)
        common_models[arm] = model
        probe_runs.append(record)
        common_init.append(record["init_sha256"])
        rows, result = evaluate_common(model, bundles[arm], child_panel, device, cfg)
        all_rows += rows
        summaries += result
        rows, result = evaluate_recursive(
            model,
            bundles[arm],
            validation,
            leaf_meta,
            leaf_vectors,
            leaf_panel,
            device,
            cfg,
        )
        all_rows += rows
        summaries += result
    if len(set(common_init)) != 1:
        raise AssertionError("common probe initialization mismatch")

    membership_init = []
    for arm in ARMS:
        print(
            f"[{time.strftime('%H:%M:%S')}] train/evaluate membership probe: {arm}",
            flush=True,
        )
        model, record = train_classifier(
            "membership", bundles[arm], schedules, train, cfg, device, out
        )
        probe_runs.append(record)
        membership_init.append(record["init_sha256"])
        rows, result = evaluate_membership(
            model, bundles[arm], validation, membership_panels, device, cfg
        )
        all_rows += rows
        summaries += result
    if len(set(membership_init)) != 1:
        raise AssertionError("membership probe initialization mismatch")

    order_init = []
    for arm in ARMS:
        print(
            f"[{time.strftime('%H:%M:%S')}] train/evaluate order probe: {arm}",
            flush=True,
        )
        model, record = train_classifier(
            "order", bundles[arm], schedules, train, cfg, device, out
        )
        probe_runs.append(record)
        order_init.append(record["init_sha256"])
        rows, result = evaluate_order(model, bundles[arm], device, cfg)
        all_rows += rows
        summaries += result
    if len(set(order_init)) != 1:
        raise AssertionError("order probe initialization mismatch")

    rows, result = evaluate_builtin_autoencoder(bundles["autoencoder"], cfg)
    all_rows += rows
    summaries += result
    controls = {arm: bundle.controls for arm, bundle in bundles.items()}
    long_status = {
        "status": "pending" if skip_long or smoke else "pending_preparation",
        "reason": "smoke run"
        if smoke
        else (
            "explicitly skipped after complete short validation"
            if skip_long
            else "run prepare_retention_long.py separately; short validation is complete"
        ),
        "training_documents_substituted": False,
    }
    run = {
        "status": "smoke_complete"
        if smoke
        else "short_validation_complete_long_pending",
        "official_test_evaluated": False,
        "model_attack_test_evaluated": False,
        "started_unix": started,
        "finished_unix": time.time(),
        "device": str(device),
        "frozen_verification": verified,
        "cache_sidecars": sidecars,
        "panel_manifest": panel_manifest,
        "metric_sanity": sanity,
        "probe_runs": probe_runs,
        "document_counts": {
            role: {
                corpus: len({source_document_id(r["id"]) for r in rows})
                for corpus, rows in values.items()
            }
            for role, values in (("train", train), ("validation", validation))
        },
        "long_validation": long_status,
    }
    atomic_json(out / "metrics.json", {"summaries": summaries, "controls": controls})
    write_csv(out / "metrics_summary.csv", summaries)
    write_csv(out / "metrics_examples.csv", all_rows)
    atomic_json(out / "RUN_STATUS.json", run)
    (out / "REPORT.md").write_text(build_report(summaries, controls, run, long_status))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files)
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Validation-only information-retention probes"
    )
    parser.add_argument(
        "--config", default="configs/information_retention_validation.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    output = run_study(
        root, config_path, device, smoke=args.smoke, skip_long=args.skip_long
    )
    print(f"completed: {output}")


if __name__ == "__main__":
    main()
