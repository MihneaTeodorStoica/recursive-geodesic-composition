"""Atomically prepare fresh v6 development safe-direct and full-document arms."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text  # noqa: E402
from protocol import (AGGREGATE_OUTPUT, CORPORA, DEVELOPMENT_SCAN, DEV_CACHE,  # noqa: E402
                      FUTURE_PROTECTED, RESERVE_BUFFER, ROOT, SAFE_RANGE,
                      TARGET_DOCUMENTS, TRAIN_SCAN, atomic_json, load_config,
                      reject_path, sha256_file, stable_hash)
from recursive_geodesic_composition.train import atomic_torch_save  # noqa: E402

CHUNK_TOKENS = 1024


def chunks(tokens):
    result = [tokens[i:i + CHUNK_TOKENS] for i in range(0, len(tokens), CHUNK_TOKENS)]
    if not result or sum(map(len, result)) != len(tokens): raise AssertionError("exact chunk accounting failed")
    return result


def depth(count):
    value = 0
    while count > 1: count = (count + 3) // 4; value += 1
    return value


def stream(corpus):
    from datasets import load_dataset
    dataset = load_dataset(corpus["dataset_name"], corpus.get("dataset_config"), split="train",
                           streaming=True, revision=corpus["dataset_revision"])
    maximum = DEVELOPMENT_SCAN[1] - DEVELOPMENT_SCAN[0]
    for offset, raw in enumerate(dataset.skip(DEVELOPMENT_SCAN[0]).take(maximum)):
        if offset >= maximum: raise AssertionError("development stream escaped declared window")
        yield DEVELOPMENT_SCAN[0] + offset, raw


def token_hash(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def fsync_tree(stage):
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle: os.fsync(handle.fileno())
    for path in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(fd)
        finally: os.close(fd)


def prepare(config_path: Path, device: str):
    cfg = load_config(config_path)
    destination = ROOT / DEV_CACHE; reject_path(destination)
    marker = destination.with_name(destination.name + ".PREPARATION_STARTED")
    frozen = ROOT / AGGREGATE_OUTPUT / "PRE_DEVELOPMENT_FROZEN.json"
    if not frozen.is_file() or not (ROOT / AGGREGATE_OUTPUT / "PRE_DEVELOPMENT_FROZEN").is_file():
        raise RuntimeError("all checkpoints/probes/floors must be frozen before development access")
    if destination.exists() or marker.exists(): raise FileExistsError("development namespace is create-only")
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"freeze_sha256={sha256_file(frozen)}\nauthorized_scan={DEVELOPMENT_SCAN}\n")
        handle.flush(); os.fsync(handle.fileno())
    stage = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}"); stage.mkdir()
    try:
        teacher = FrozenTeacher(cfg["model_name"], cfg["model_revision"], device)
        if int(getattr(teacher.model.config, "max_position_embeddings", 0)) - 2 < SAFE_RANGE[1]:
            raise ValueError("teacher context is below safe direct target bound")
        summaries, artifacts = {}, {}
        for corpus_cfg in cfg["corpora"]:
            corpus = corpus_cfg["id"]; safe_rows, index_rows = [], []
            counters = {"scanned": 0, "accepted": 0, "blank": 0, "too_short": 0}
            last = DEVELOPMENT_SCAN[0] - 1
            for source_index, raw in stream(corpus_cfg):
                counters["scanned"] += 1; last = source_index
                text = raw.get("text") or raw.get("content") or ""
                if not isinstance(text, str) or not text.strip(): counters["blank"] += 1; continue
                token_ids = tokenize_text(teacher.tokenizer, text)
                if len(token_ids) < SAFE_RANGE[0]: counters["too_short"] += 1; continue
                # Eligibility is final. Operational failures are fatal, never skips.
                prefix_count = min(len(token_ids), SAFE_RANGE[1]); full_chunks = chunks(token_ids)
                embeddings = []
                for start in range(0, len(full_chunks), 16):
                    part = teacher.embed(full_chunks[start:start + 16], 16).float()
                    if not torch.isfinite(part).all(): raise FloatingPointError("nonfinite full leaf")
                    embeddings.append(part)
                full_leaves = torch.cat(embeddings); full_lengths = [len(x) for x in full_chunks]
                safe_leaf_count = (prefix_count + CHUNK_TOKENS - 1) // CHUNK_TOKENS
                safe_ids = token_ids[:prefix_count]
                target = teacher.embed([safe_ids], 1)[0].float()
                if not torch.isfinite(target).all(): raise FloatingPointError("nonfinite safe target")
                rank = len(safe_rows); row_id = f"{corpus}:{source_index}"
                shard_relative = f"{corpus}/full/{rank:06d}.pt"; shard_path = stage / shard_relative
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                shard = {"id": row_id, "corpus": corpus, "source_index": source_index,
                         "source_document_id": str(raw.get("id", source_index)),
                         "role": "development", "arm": "unrestricted_full_document_diagnostic",
                         "arity": 4, "chunk_tokens": CHUNK_TOKENS,
                         "truncated": False, "token_cap": None, "maximum_leaves": None,
                         "recursion_depth_cap": None, "full_token_count": len(token_ids),
                         "full_leaf_count": len(full_chunks), "tree_depth": depth(len(full_chunks)),
                         "token_ids_sha256": token_hash(token_ids),
                         "leaves": full_leaves.half(), "lengths": torch.tensor(full_lengths, dtype=torch.int32)}
                atomic_torch_save(shard, shard_path)
                index_rows.append({key: shard[key] for key in ("id", "corpus", "source_index", "source_document_id", "full_token_count", "full_leaf_count", "tree_depth", "token_ids_sha256")} | {
                    "selected_rank": rank, "shard": shard_relative, "shard_sha256": sha256_file(shard_path), "shard_bytes": shard_path.stat().st_size})
                safe_lengths = full_lengths[:safe_leaf_count]
                if sum(safe_lengths) != prefix_count: raise AssertionError("safe/full prefix mismatch")
                safe_rows.append({"id": row_id, "corpus": corpus, "source_index": source_index,
                                  "source_document_id": shard["source_document_id"], "selected_rank": rank,
                                  "role": "development", "arm": "safe_direct_k4", "arity": 4,
                                  "reduction": "direct_single_k4", "safe_prefix_token_count": prefix_count,
                                  "full_token_count": len(token_ids), "token_ids_sha256": shard["token_ids_sha256"],
                                  "leaves": full_leaves[:safe_leaf_count].clone(),
                                  "lengths": torch.tensor(safe_lengths, dtype=torch.float32), "teacher_target": target})
                counters["accepted"] += 1
                if len(safe_rows) == TARGET_DOCUMENTS: break
            if len(safe_rows) != TARGET_DOCUMENTS:
                raise RuntimeError(f"{corpus}: insufficient eligible rows in {DEVELOPMENT_SCAN}")
            if last >= DEVELOPMENT_SCAN[1] or counters["scanned"] != last + 1 - DEVELOPMENT_SCAN[0]:
                raise AssertionError("scan accounting escaped development range")
            directory = stage / corpus; directory.mkdir(exist_ok=True)
            safe_path, index_path = directory / "safe.pt", directory / "full_index.json"
            atomic_torch_save(safe_rows, safe_path)
            atomic_json(index_path, {"role": "development", "corpus": corpus, "documents": index_rows})
            summaries[corpus] = {**counters, "actual_scanned_range": [DEVELOPMENT_SCAN[0], last + 1],
                                 "selection": "first_1000_safe_direct_length_eligible_in_source_order",
                                 "source_indices_sha256": stable_hash([r["source_index"] for r in safe_rows]),
                                 "document_order_sha256": stable_hash([r["id"] for r in safe_rows]),
                                 "safe_sha256": sha256_file(safe_path), "index_sha256": sha256_file(index_path),
                                 "full_shards": len(index_rows), "full_shard_bytes": sum(r["shard_bytes"] for r in index_rows)}
            artifacts[f"{corpus}/safe.pt"] = summaries[corpus]["safe_sha256"]
            artifacts[f"{corpus}/full_index.json"] = summaries[corpus]["index_sha256"]
            for row in index_rows: artifacts[row["shard"]] = row["shard_sha256"]
            print(corpus, summaries[corpus], flush=True)
        manifest = {"study": cfg["study"].rsplit("_seed", 1)[0], "cache_namespace": DEV_CACHE,
                    "role": "development", "development_only": True, "protected_data_accessed": False,
                    "source_scan": list(DEVELOPMENT_SCAN), "train_scan_refused": list(TRAIN_SCAN),
                    "reserve_buffer_untouched": list(RESERVE_BUFFER), "future_protected_untouched": list(FUTURE_PROTECTED),
                    "maximum_source_index_opened_exclusive": max(v["actual_scanned_range"][1] for v in summaries.values()),
                    "selection": "first_1000_safe_direct_length_eligible_in_source_order",
                    "safe_direct": {"tokens": list(SAFE_RANGE), "arity": 4, "documents_per_corpus": TARGET_DOCUMENTS},
                    "full_document_diagnostic": {"same_documents": True, "every_available_token": True,
                                                 "token_cap": None, "maximum_leaves": None, "truncation": False,
                                                 "teacher_comparison": False, "auc": False},
                    "corpora": list(CORPORA), "summaries": summaries, "artifacts": artifacts,
                    "pre_development_freeze_sha256": sha256_file(frozen), "preparer_sha256": sha256_file(__file__)}
        atomic_json(stage / "CACHE_MANIFEST.json", manifest)
        (stage / "COMPLETE").write_text(f"{sha256_file(stage / 'CACHE_MANIFEST.json')}  CACHE_MANIFEST.json\n")
        fsync_tree(stage); os.replace(stage, destination)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(ROOT / "configs/final_v6/seed424242.yaml")); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); print(prepare(Path(args.config).resolve(), args.device))
