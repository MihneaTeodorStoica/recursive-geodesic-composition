"""Prepare a separate provenance-complete K-ary train/validation cache.

This module has no test/official role.  Targets are direct teacher embeddings of
exact contiguous token spans and are present only at <= 6144 tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
import yaml

from .architecture import contiguous_reduction_groups
from .data import FrozenTeacher, stream_document_ranges, tokenize_text, chunk_token_ids
from .prepare import (
    _embed_targets,
    _random_seed,
    _training_segmentations,
    random_chunk_token_ids,
)

CACHE_NAMESPACE = "kary_geodesic_v1"
SAFE_TEACHER_MAX = 6144
FORBIDDEN_ROLE_WORDS = ("test", "official")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_free_storage(path, minimum_bytes, context):
    """Fail before a large immutable-cache/checkpoint operation."""
    minimum = int(minimum_bytes or 0)
    if minimum < 0:
        raise ValueError("minimum free storage must be non-negative")
    free = shutil.disk_usage(Path(path).resolve()).free
    if free < minimum:
        raise RuntimeError(
            f"storage guard failed before {context}: free={free} required={minimum}")
    return free


def _reject_forbidden(value):
    lowered = str(value).lower()
    if any(word in lowered for word in FORBIDDEN_ROLE_WORDS):
        raise ValueError(f"refusing forbidden test/official value: {value}")


def kary_target_spans(chunks, arity, teacher_max_tokens=SAFE_TEACHER_MAX):
    """Return direct merged spans and auditable serialized target provenance."""
    arity = int(arity)
    safe = int(teacher_max_tokens)
    if arity < 2:
        raise ValueError("arity must be >= 2")
    if safe < 1 or safe > SAFE_TEACHER_MAX:
        raise ValueError("teacher_max_tokens must be in 1..6144")
    if len(chunks) < 2 or any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("a K-ary row requires at least two non-empty chunks")

    offsets, cursor = [], 0
    for chunk in chunks:
        offsets.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)
    nodes = [
        {
            "tokens": list(chunk),
            "leaf_start": index,
            "leaf_end": index + 1,
            "token_start": offsets[index][0],
            "token_end": offsets[index][1],
        }
        for index, chunk in enumerate(chunks)
    ]
    spans, provenance, level = [], [], 0
    while len(nodes) > 1:
        next_nodes = []
        for group_index, (start, width) in enumerate(
            contiguous_reduction_groups(len(nodes), arity)
        ):
            group = nodes[start : start + width]
            if width == 1:
                next_nodes.append(group[0])
                continue
            tokens = [token for node in group for token in node["tokens"]]
            parent = {
                "tokens": tokens,
                "leaf_start": group[0]["leaf_start"],
                "leaf_end": group[-1]["leaf_end"],
                "token_start": group[0]["token_start"],
                "token_end": group[-1]["token_end"],
            }
            token_count = len(tokens)
            target_index = len(spans)
            spans.append(tokens)
            provenance.append(
                {
                    "target_index": target_index,
                    "level": level,
                    "group_index": group_index,
                    "group_arity": width,
                    "leaf_start": parent["leaf_start"],
                    "leaf_end": parent["leaf_end"],
                    "token_start": parent["token_start"],
                    "token_end": parent["token_end"],
                    "token_count": token_count,
                    "teacher_eligible": token_count <= safe,
                    "span_token_sha256": hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
            next_nodes.append(parent)
        nodes = next_nodes
        level += 1
    return spans, provenance


def make_kary_row(doc_id, chunks, teacher, cfg, *, segmentation="train"):
    """Build one true K-ary row; never infer targets from binary target order."""
    arity = int(cfg["arity"])
    maximum_leaves = cfg.get("train_max_tree_leaves") if segmentation == "train" else cfg.get("validation_max_tree_leaves")
    if maximum_leaves is not None:
        chunks = chunks[: int(maximum_leaves)]
    if len(chunks) < 2:
        return None
    leaves = teacher.embed(chunks, int(cfg.get("leaf_batch_size", 32)))
    if not torch.isfinite(leaves).all():
        raise FloatingPointError(f"non-finite leaf embedding for {doc_id}")
    expected_dimension = cfg.get("output_dimension")
    if expected_dimension is not None and leaves.shape[-1] != int(expected_dimension):
        raise ValueError(
            f"teacher output dimension mismatch for {doc_id}: "
            f"expected={expected_dimension} actual={leaves.shape[-1]}")
    spans, provenance = kary_target_spans(
        chunks, arity, int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX))
    )
    targets = _embed_targets(
        teacher,
        spans,
        int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX)),
        int(cfg.get("teacher_target_batch_size", 32)),
        int(cfg.get("teacher_target_max_padded_tokens", 32768)),
        context=f"kary doc_id={doc_id} arity={arity} segmentation={segmentation}",
    )
    if any((target is not None) != item["teacher_eligible"]
           for target, item in zip(targets, provenance)):
        raise AssertionError("teacher eligibility and generated targets diverged")
    return {
        "id": str(doc_id),
        "segmentation": str(segmentation),
        "cache_namespace": CACHE_NAMESPACE,
        "arity": arity,
        "reduction": "contiguous_balanced",
        "teacher_max_tokens": int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX)),
        "leaves": leaves,
        "lengths": torch.tensor([len(chunk) for chunk in chunks], dtype=torch.float32),
        "targets": targets,
        "target_provenance": provenance,
    }


def _segmentations_for_role(cfg, document, token_ids, role):
    if role == "train":
        return [("train", chunks) for chunks in _training_segmentations(
            cfg, document, token_ids, int(cfg.get("train_augmentations", 1))
        )]

    # Every validation segmentation covers the exact same safe prefix, making
    # root teacher targets and cross-segmentation consistency directly matched.
    safe_ids = token_ids[: int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX))]
    fixed = chunk_token_ids(safe_ids, int(cfg["chunk_tokens"]))
    # The standard chunker intentionally drops a 1--3 token tail. Make that
    # exact covered prefix the common span for every validation segmentation.
    safe_ids = safe_ids[:sum(map(len, fixed))]
    result = [("fixed", fixed)]
    if int(cfg.get("validation_segmentations", 1)) > 1:
        rng = random.Random(_random_seed(cfg.get("seed", 0), document.id, 0, "kary-validation-chunks"))
        alternate = random_chunk_token_ids(
            safe_ids,
            int(cfg.get("random_chunk_min", 256)),
            int(cfg.get("random_chunk_max", cfg["chunk_tokens"])),
            rng,
        )
        covered = sum(map(len, alternate))
        if alternate and covered < len(safe_ids):
            alternate[-1] = alternate[-1] + safe_ids[covered:]
        result.append(("alternate", alternate))
    return result[: int(cfg.get("validation_segmentations", 1))]


def prepare_kary_split(cfg, teacher, corpus, role, documents):
    if role not in ("train", "validation"):
        raise ValueError("K-ary preparer supports only train/validation roles")
    rows = []
    corpus_id = str(corpus["id"])
    for document in documents:
        doc_id = f"{corpus_id}:{document.id}"
        token_ids = tokenize_text(teacher.tokenizer, document.text)
        for segmentation, chunks in _segmentations_for_role(cfg, document, token_ids, role):
            row = make_kary_row(doc_id, chunks, teacher, cfg,
                                segmentation="train" if role == "train" else segmentation)
            if row is not None:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no usable K-ary rows for {corpus_id}/{role}")
    destination = Path(cfg["cache_dir"]) / corpus_id / f"{role}.pt"
    _reject_forbidden(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, destination)
    sidecar = {
        "cache_namespace": CACHE_NAMESPACE,
        "split": role,
        "validation_only_experiment": True,
        "official_test_evaluated": False,
        "arity": int(cfg["arity"]),
        "reduction": "contiguous_balanced",
        "teacher_max_tokens": int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX)),
        "target_provenance": "per-target exact leaf/token interval and token SHA-256",
        "source_start": int(corpus[f"{role}_start"]),
        "requested": int(corpus[f"{role}_documents"]),
        "usable": len(rows),
        "corpus_id": corpus_id,
        "dataset": corpus["dataset_name"],
        "config": corpus.get("dataset_config"),
        "revision": corpus["dataset_revision"],
        "model": cfg["model_name"],
        "model_revision": cfg["model_revision"],
        "teacher_pooling": cfg.get("teacher_pooling", "masked_mean"),
        "teacher_provenance": teacher.provenance() if hasattr(teacher, "provenance") else None,
    }
    Path(str(destination) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"{corpus_id}/{role}: {len(rows)} -> {destination}", flush=True)


def prepare_kary(cfg, device="cuda", smoke=False):
    if cfg.get("validation_only") is not True or cfg.get("official_test_evaluated") is not False:
        raise ValueError("K-ary config must be explicitly validation-only and unofficial")
    if int(cfg.get("arity", 0)) < 2:
        raise ValueError("arity must be >= 2")
    if int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX)) > SAFE_TEACHER_MAX:
        raise ValueError("teacher supervision above 6144 tokens is forbidden")
    _reject_forbidden(cfg["cache_dir"])
    cache_root = Path(cfg["cache_dir"])
    if bool(cfg.get("refuse_overwrite_cache", False)) and (cache_root / "CACHE_MANIFEST.json").exists():
        raise FileExistsError(f"immutable K-ary cache is already complete: {cache_root}")
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    require_free_storage(cache_root.parent,
                         cfg.get("minimum_free_bytes_before_prepare", 0),
                         "K-ary cache preparation")
    corpora = cfg.get("corpora", [])
    if not corpora or abs(sum(float(c["weight"]) for c in corpora) - 1.0) > 1e-8:
        raise ValueError("positive corpus weights must sum to one")
    if any(float(c["weight"]) <= 0 for c in corpora):
        raise ValueError("corpus weights must be positive")
    teacher = FrozenTeacher(
        cfg["model_name"], cfg["model_revision"], device,
        pooling=cfg.get("teacher_pooling", "masked_mean"),
        trust_remote_code=cfg.get("teacher_trust_remote_code"))
    for corpus in corpora:
        train_range = (int(corpus["train_start"]), int(corpus["train_documents"]))
        validation_range = (int(corpus["validation_start"]), int(corpus["validation_documents"]))
        if max(train_range[0], validation_range[0]) < min(sum(train_range), sum(validation_range)):
            raise ValueError(f"overlapping train/validation source ranges for {corpus['id']}")
        ranges = {"train": train_range, "validation": validation_range}
        if smoke:
            ranges = {"train": (train_range[0], 4), "validation": (validation_range[0], 4)}
        documents = stream_document_ranges(
            corpus["dataset_name"], corpus.get("dataset_config"),
            corpus["dataset_revision"], ranges
        )
        for role in ("train", "validation"):
            prepare_kary_split(cfg, teacher, corpus, role, documents[role])
    artifacts = {}
    for path in sorted(cache_root.glob("*/*")):
        if path.is_file():
            artifacts[str(path.relative_to(cache_root))] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "cache_namespace": CACHE_NAMESPACE,
        "validation_only": True,
        "official_test_evaluated": False,
        "arity": int(cfg["arity"]),
        "fixed_chunk_tokens": int(cfg["chunk_tokens"]),
        "random_chunk_min": int(cfg["random_chunk_min"]),
        "random_chunk_max": int(cfg["random_chunk_max"]),
        "teacher_max_tokens": int(cfg.get("teacher_max_tokens", SAFE_TEACHER_MAX)),
        "teacher_pooling": cfg.get("teacher_pooling", "masked_mean"),
        "teacher_provenance": teacher.provenance() if hasattr(teacher, "provenance") else None,
        "full_first_level_tokens": int(cfg["arity"]) * int(cfg["chunk_tokens"]),
        "config_sha256": hashlib.sha256(
            yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest(),
        "artifacts": artifacts,
    }
    (cache_root / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare true K-ary train/validation cache")
    parser.add_argument("--config", default="configs/kary4_geodesic_encoder.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    prepare_kary(cfg, args.device, args.smoke)


if __name__ == "__main__":
    main()
