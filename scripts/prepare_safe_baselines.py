#!/usr/bin/env python3
"""Regenerate safe-only teacher-comparable tensors in a new, resumable namespace.

No merger inference, fitting, or metric computation occurs here. The original
cache, official markers, and full-document protocol are never modified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from safe_baselines_protocol import (
    DEFAULT_CONFIG, ROOT, PUBLISHED_HASH, atomic_json, atomic_tensor, configure_device,
    environment, load_protocol, require_hash, sha256, source_hashes, stable_hash,
    validate_population,
)


def token_hash(ids):
    # Matches the original official preparation's token-identity digest.
    import hashlib
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def select_rows(corpus_cfg, tokenizer, official, published):
    from datasets import load_dataset
    from recursive_geodesic_composition.data import tokenize_text

    corpus = corpus_cfg["id"]
    start, end = int(official["scan_start"]), int(official["scan_end_exclusive"])
    dataset = load_dataset(corpus_cfg["dataset_name"], corpus_cfg.get("dataset_config"),
                          split="train", streaming=True, revision=corpus_cfg["dataset_revision"])
    rows = []
    for offset, raw in enumerate(dataset.skip(start).take(end - start)):
        source_index = start + offset
        text = raw.get("text") or raw.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = tokenize_text(tokenizer, text)
        if len(tokens) < 1025:
            continue
        safe = tokens[:4096]
        rows.append({
            "id": f"{corpus}:{source_index}", "corpus": corpus,
            "source_index": source_index, "source_document_id": str(raw.get("id", source_index)),
            "selected_rank": len(rows), "full_token_count": len(tokens),
            "safe_prefix_token_count": len(safe), "safe_token_ids": safe,
            "safe_prefix_token_ids_sha256": token_hash(safe),
            "full_token_ids_sha256": token_hash(tokens),
        })
        if len(rows) % 100 == 0:
            print(f"{corpus}: selected {len(rows)}/1000, source index {source_index}", flush=True)
        if len(rows) == 1000:
            break
    validate_population(rows, corpus, published)
    return rows


def prepare(config_path, device_name, select_only=False):
    cfg, official, published = load_protocol(config_path)
    device = configure_device(cfg, device_name)
    cache = ROOT / cfg["cache_dir"]
    if (cache / "COMPLETE").exists():
        from safe_baselines_protocol import load_safe_cache
        load_safe_cache(cfg, published)
        print("Complete cache already exists and verifies; no regeneration.", flush=True)
        return
    cache.mkdir(exist_ok=True)
    freeze_path = cache / "PREPARATION_PROTOCOL.json"
    # Pin the configuration before the first corpus access. Resumes cannot change
    # selection, representation, or evaluation settings after observing data.
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        if freeze["config_sha256"] != cfg["config_sha256"]:
            raise ValueError("preparation configuration changed across resume")
        for name, digest in freeze["preparation_source_hashes"].items():
            require_hash(ROOT / name, digest)
    else:
        hashes = source_hashes()
        preparation_names = ("scripts/prepare_safe_baselines.py", "scripts/safe_baselines_protocol.py",
                             "src/recursive_geodesic_composition/data.py", "configs/official_v6.yaml")
        freeze = {
            "config_sha256": cfg["config_sha256"], "configuration": cfg,
            "started_unix": time.time(), "environment": environment(device),
            "preparation_source_hashes": {key: hashes[key] for key in preparation_names},
            "published_results_sha256": PUBLISHED_HASH,
            "cache_kind": "regenerated_safe_only_not_original_cache",
            "tokenization": "one document tokenization; add_special_tokens=False; truncation=False",
            "representation": "masked mean of final hidden states then L2 normalization",
            "batching_difference": "only safe-prefix leaves, up to four per document; original embedded full-document leaves in batches up to 16",
        }
        atomic_json(freeze_path, freeze)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(official["model_name"], revision=official["model_revision"])
    token_rows = {}
    for corpus_cfg in official["corpora"]:
        corpus = corpus_cfg["id"]
        path = cache / corpus / "selected_tokens.pt"
        sidecar = path.with_suffix(".json")
        if path.exists():
            metadata = json.loads(sidecar.read_text())
            if metadata["config_sha256"] != cfg["config_sha256"]:
                raise ValueError("token cache configuration changed")
            require_hash(path, metadata["sha256"])
            rows = torch.load(path, map_location="cpu", weights_only=True)
            validate_population(rows, corpus, published)
        else:
            print(f"Selecting {corpus} from pinned source...", flush=True)
            rows = select_rows(corpus_cfg, tokenizer, official, published)
            atomic_tensor(path, rows)
            atomic_json(sidecar, {"config_sha256": cfg["config_sha256"], "sha256": sha256(path),
                                  "document_order_sha256": stable_hash([row["id"] for row in rows])})
        token_rows[corpus] = rows
    if select_only:
        print("All 4,000 selected document identities and prefix token totals match published records.", flush=True)
        return

    from recursive_geodesic_composition.data import FrozenTeacher
    teacher = FrozenTeacher(official["model_name"], official["model_revision"], str(device))
    if teacher.model.config.max_position_embeddings - 2 < 4096:
        raise ValueError("teacher cannot encode the specified prefix without truncation")
    if teacher.model.config.hidden_size != 768:
        raise ValueError("teacher dimension changed")
    # Preserve the original FrozenTeacher loader: this pinned model defaults to
    # FP16 parameters; its normalized outputs are explicitly stored as FP32.
    # Forcing FP32 weights here would change the original representation procedure.
    expected_dtype = getattr(torch, cfg["preparation"]["teacher_parameter_dtype"])
    parameter_dtypes = {str(parameter.dtype) for parameter in teacher.model.parameters()}
    if parameter_dtypes != {str(expected_dtype)}:
        raise ValueError(f"teacher parameter dtype changed: {parameter_dtypes}")
    summaries = {}
    for corpus_cfg in official["corpora"]:
        corpus = corpus_cfg["id"]
        target_path = cache / corpus / "safe.pt"
        sidecar = target_path.with_suffix(".json")
        if target_path.exists():
            summaries[corpus] = json.loads(sidecar.read_text())
            require_hash(target_path, summaries[corpus]["safe_sha256"])
            continue
        rows, started = [], time.perf_counter()
        for record in token_rows[corpus]:
            token_ids = record["safe_token_ids"]
            chunks = [token_ids[index:index + 1024] for index in range(0, len(token_ids), 1024)]
            leaves = teacher.embed(chunks, int(cfg["preparation"]["leaf_batch_size"])).float()
            target = teacher.embed([token_ids], int(cfg["preparation"]["target_batch_size"]))[0].float()
            if not torch.isfinite(leaves).all() or not torch.isfinite(target).all():
                raise FloatingPointError(f"nonfinite teacher output for {record['id']}; no skipping allowed")
            rows.append({key: value for key, value in record.items() if key != "safe_token_ids"} | {
                "leaves": leaves.cpu(), "lengths": torch.tensor([len(chunk) for chunk in chunks], dtype=torch.float32),
                "teacher_target": target.cpu(),
            })
            if len(rows) % 100 == 0:
                print(f"{corpus}: embedded {len(rows)}/1000 ({time.perf_counter() - started:.1f}s)", flush=True)
        validate_population(rows, corpus, published)
        atomic_tensor(target_path, rows)
        summaries[corpus] = {
            "documents": len(rows), "safe_sha256": sha256(target_path),
            "selected_tokens_sha256": sha256(cache / corpus / "selected_tokens.pt"),
            "document_order_sha256": stable_hash([row["id"] for row in rows]),
            "embedding_seconds": time.perf_counter() - started,
        }
        atomic_json(sidecar, summaries[corpus])
    manifest = {
        **freeze, "completed_unix": time.time(), "corpora": summaries,
        "pooled_document_order_sha256": stable_hash([row["id"] for corpus in published["corpus_order"] for row in token_rows[corpus]]),
        "teacher_name": official["model_name"], "teacher_revision": official["model_revision"],
        "datasets": official["corpora"], "original_tensor_identity_claimed": False,
        "teacher_parameter_dtypes": sorted(parameter_dtypes),
        "embedding_storage_dtype": "float32", "population_identity_verified": True,
    }
    atomic_json(cache / "CACHE_MANIFEST.json", manifest)
    (cache / "COMPLETE").write_text(sha256(cache / "CACHE_MANIFEST.json") + "\n")
    from safe_baselines_protocol import load_safe_cache
    load_safe_cache(cfg, published)
    print(f"Verified regenerated safe cache: {cache}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.device, args.select_only)
