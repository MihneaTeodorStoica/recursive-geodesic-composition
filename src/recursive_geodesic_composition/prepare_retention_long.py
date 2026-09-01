"""Prepare long held-out retention leaves from pinned validation ranges only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import FrozenTeacher, chunk_token_ids, stream_document_ranges, tokenize_text
from .information_retention import (
    CORPORA,
    atomic_json,
    load_protocol,
    sha256_file,
    validate_long_rows,
)


def prepare(config: Path, device: str = "cuda") -> dict:
    config = Path(config).resolve()
    root = config.parent.parent
    cfg, source, verified = load_protocol(root, config)
    long_cfg = cfg["long_validation"]
    if int(long_cfg["source_start"]) != 12000 or int(long_cfg["source_count"]) != 1000:
        raise ValueError(
            "long preparer may use only pinned validation indices [12000,13000)"
        )
    cache = root / long_cfg["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    teacher = FrozenTeacher(source["model_name"], source["model_revision"], device)
    source_by = {item["id"]: item for item in source["corpora"]}
    manifest = {
        "validation_only": True,
        "official_test_evaluated": False,
        "source_start": 12000,
        "source_count": 1000,
        "model": source["model_name"],
        "model_revision": source["model_revision"],
        "frozen_checkpoint_verification": verified,
        "corpora": {},
    }
    for corpus in CORPORA:
        destination = cache / f"{corpus}.pt"
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite long validation shard: {destination}"
            )
        pinned = source_by[corpus]
        documents = stream_document_ranges(
            pinned["dataset_name"],
            pinned.get("dataset_config"),
            pinned["dataset_revision"],
            {"validation": (12000, 1000)},
        )["validation"]
        selected = {bucket["name"]: [] for bucket in long_cfg["token_buckets"]}
        selected_ids = set()
        for document in documents:
            token_ids = tokenize_text(teacher.tokenizer, document.text)
            for bucket in long_cfg["token_buckets"]:
                name = bucket["name"]
                if (
                    len(selected[name]) < int(bucket["documents"])
                    and int(bucket["min"]) <= len(token_ids) <= int(bucket["max"])
                    and document.id not in selected_ids
                ):
                    selected[name].append((document.id, token_ids[:32768]))
                    selected_ids.add(document.id)
                    break
            if all(
                len(selected[b["name"]]) >= int(b["documents"])
                for b in long_cfg["token_buckets"]
            ):
                break
        rows = []
        for bucket in long_cfg["token_buckets"]:
            for document_id, token_ids in selected[bucket["name"]]:
                chunks = chunk_token_ids(token_ids, int(source["chunk_tokens"]))
                leaves = teacher.embed(
                    chunks, int(source.get("teacher_target_batch_size", 32))
                ).half()
                row = {
                    "id": f"{corpus}:{document_id}",
                    "corpus": corpus,
                    "leaves": leaves,
                    "lengths": torch.tensor(
                        [len(x) for x in chunks], dtype=torch.int16
                    ),
                    "token_count": len(token_ids),
                    "token_bucket": bucket["name"],
                    "source_start": 12000,
                    "source_count": 1000,
                    "dataset_revision": pinned["dataset_revision"],
                    "model_revision": source["model_revision"],
                }
                rows.append(row)
        rows.sort(key=lambda row: (row["token_bucket"], row["id"]))
        validate_long_rows(rows)
        if len(rows) > int(long_cfg["max_documents_per_corpus"]):
            raise ValueError("long shard exceeds document cap")
        tensor_bytes = sum(
            row["leaves"].numel() * row["leaves"].element_size()
            + row["lengths"].numel() * row["lengths"].element_size()
            for row in rows
        )
        if tensor_bytes > int(long_cfg["max_shard_bytes"]):
            raise ValueError(
                f"long shard exceeds size cap before serialization: {tensor_bytes}"
            )
        torch.save(rows, destination)
        sidecar = {
            "split": "retention_validation_long",
            "official_test_evaluated": False,
            "corpus_id": corpus,
            "source_start": 12000,
            "source_count": 1000,
            "dataset": pinned["dataset_name"],
            "config": pinned.get("dataset_config"),
            "dataset_revision": pinned["dataset_revision"],
            "model": source["model_name"],
            "model_revision": source["model_revision"],
            "documents": len(rows),
            "by_bucket": {name: len(values) for name, values in selected.items()},
            "tensor_bytes": tensor_bytes,
            "contains_text": False,
            "contains_full_span_teacher_target": False,
            "shard_sha256": sha256_file(destination),
        }
        atomic_json(Path(str(destination) + ".json"), sidecar)
        manifest["corpora"][corpus] = sidecar
        print(
            f"{corpus}: {len(rows)} long validation documents -> {destination}",
            flush=True,
        )
    atomic_json(cache / "MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pinned long retention-validation leaves"
    )
    parser.add_argument(
        "--config", default="configs/information_retention_validation.yaml"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = prepare(Path(args.config).resolve(), args.device)
    print(
        json.dumps(
            {c: x["documents"] for c, x in manifest["corpora"].items()}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
