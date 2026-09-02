#!/usr/bin/env python3
"""Create the immutable, shared downstream leaf cache (no model evaluation)."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time, uuid
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
OUTPUT = ROOT / "results/v3_cls/downstream/leaf_cache"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rows_sha(ids, labels, token_counts) -> str:
    payload = [{"id": x, "labels": y, "raw_token_count": n} for x, y, n in zip(ids, labels, token_counts)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_rows(key: str, split: str, spec: dict):
    ds = load_dataset(spec["id"], spec["config"], split=split, revision=spec["revision"])
    rows = []
    if key == "ecthr_a":
        for i, row in enumerate(ds):
            paragraphs = row["text"]
            if not isinstance(paragraphs, list) or not paragraphs or any(not isinstance(p, str) for p in paragraphs):
                raise ValueError(f"ecthr_a/{split}/{i}: malformed paragraph list (no drops permitted)")
            text = "\n\n".join(paragraphs)
            stored = [int(x) for x in row["labels"]]
            if len(set(stored)) != len(stored) or any(x < 0 or x >= 10 for x in stored):
                raise ValueError(f"ecthr_a/{split}/{i}: malformed labels")
            labels = sorted(stored) + ([10] if not stored else [])
            rows.append((f"{split}:{i}", text, labels))
    else:
        for i, row in enumerate(ds):
            text, label = row["text"], int(row["label"])
            # The canonical cleaned corpus contains genuinely empty documents.
            # Preserve them; they are represented by the teacher's CLS/SEP-only
            # empty sequence rather than silently removed.
            if not isinstance(text, str) or not 0 <= label < 20:
                raise ValueError(f"20_newsgroups/{split}/{i}: malformed row (no drops permitted)")
            rows.append((f"{split}:{i}", text, label))
    if len(rows) != len(ds):
        raise AssertionError("row-count mismatch")
    return rows, ds._fingerprint, ds.features


def prepare_split(teacher: FrozenTeacher, rows, out: Path, *, leaf_batch: int, direct_batch: int):
    all_leaves, all_lengths, offsets = [], [], [0]
    direct = torch.full((len(rows), teacher.model.config.hidden_size), float("nan"), dtype=torch.float32)
    direct_valid = torch.zeros(len(rows), dtype=torch.bool)
    ids, labels, token_counts = [], [], []
    pending_chunks, pending_lengths, pending_docs = [], [], []
    started = time.perf_counter()

    def flush_leaves():
        if not pending_docs:
            return
        embedded = teacher.embed(pending_chunks, leaf_batch).float()
        cursor = 0
        for doc_index, lengths in zip(pending_docs, pending_lengths):
            count = len(lengths)
            if doc_index != len(all_leaves):
                raise AssertionError("leaf order changed")
            all_leaves.append(embedded[cursor:cursor + count].clone())
            all_lengths.append(torch.tensor(lengths, dtype=torch.int32))
            offsets.append(offsets[-1] + count)
            cursor += count
        pending_chunks.clear(); pending_lengths.clear(); pending_docs.clear()

    pending_direct_ids, pending_direct_indexes = [], []
    def flush_direct():
        if not pending_direct_ids:
            return
        values = teacher.embed(pending_direct_ids, direct_batch).float()
        direct[pending_direct_indexes] = values
        direct_valid[pending_direct_indexes] = True
        pending_direct_ids.clear(); pending_direct_indexes.clear()

    context_raw_limit = int(teacher.model.config.max_position_embeddings) - 2
    for index, (row_id, text, row_labels) in enumerate(rows):
        token_ids = tokenize_text(teacher.tokenizer, text)
        # Empty cleaned 20NG rows remain one empty-sequence CLS leaf with zero
        # raw-token mass. This keeps all canonical rows and is explicit in the
        # manifest; nonempty documents retain exact contiguous 1024-token chunks.
        chunks = [token_ids[i:i + 1024] for i in range(0, len(token_ids), 1024)] or [[]]
        pending_chunks.extend(chunks)
        pending_lengths.append([len(x) for x in chunks])
        pending_docs.append(index)
        ids.append(row_id); labels.append(row_labels); token_counts.append(len(token_ids))
        if len(token_ids) <= context_raw_limit:
            pending_direct_ids.append(token_ids); pending_direct_indexes.append(index)
        if sum(map(len, pending_lengths)) >= 256:
            flush_leaves()
        if len(pending_direct_ids) >= 16 or (pending_direct_ids and max(map(len, pending_direct_ids)) > 4096):
            flush_direct()
        if (index + 1) % 500 == 0:
            print(f"{out.parent.name}/{out.stem}: {index + 1}/{len(rows)} rows, {time.perf_counter()-started:.1f}s", flush=True)
    flush_leaves(); flush_direct()
    leaves = torch.cat(all_leaves, 0).contiguous()
    lengths = torch.cat(all_lengths, 0).contiguous()
    offsets_tensor = torch.tensor(offsets, dtype=torch.int64)
    if len(offsets) != len(rows) + 1 or offsets[-1] != len(leaves) or int(lengths.sum()) != sum(token_counts):
        raise AssertionError("exact token/offset accounting failed")
    artifact = {
        "schema": "v3_cls_downstream_leaf_cache_split_v1", "ids": ids, "labels": labels,
        "raw_token_counts": torch.tensor(token_counts, dtype=torch.int64), "offsets": offsets_tensor,
        "leaves": leaves, "lengths": lengths, "direct_teacher": direct, "direct_valid": direct_valid,
        "chunk_tokens": 1024, "rows_sha256": rows_sha(ids, labels, token_counts),
        "leaves_tensor_sha256": tensor_sha(leaves), "lengths_tensor_sha256": tensor_sha(lengths),
    }
    torch.save(artifact, out)
    return {
        "rows": len(rows), "leaves": len(leaves), "raw_tokens": int(lengths.sum()),
        "minimum_tokens": min(token_counts), "maximum_tokens": max(token_counts),
        "direct_valid": int(direct_valid.sum()), "direct_invalid": int((~direct_valid).sum()),
        "rows_sha256": artifact["rows_sha256"], "leaves_tensor_sha256": artifact["leaves_tensor_sha256"],
        "lengths_tensor_sha256": artifact["lengths_tensor_sha256"], "artifact_sha256": file_sha(out),
        "bytes": out.stat().st_size, "seconds": time.perf_counter() - started,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--leaf-batch-size", type=int, default=32)
    ap.add_argument("--direct-batch-size", type=int, default=2)
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if OUTPUT.exists():
        raise FileExistsError(f"create-only cache already exists: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = OUTPUT.with_name(f".{OUTPUT.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir()
    overall = time.perf_counter()
    try:
        t = protocol["teacher"]
        teacher = FrozenTeacher(t["id"], t["revision"], args.device, pooling="cls", trust_remote_code=True)
        if teacher.provenance()["max_position_embeddings"] != 8192 or teacher.provenance()["hidden_size"] != 768:
            raise ValueError("unexpected pinned teacher dimensions")
        manifest = {
            "schema": "v3_cls_downstream_leaf_cache_manifest_v1", "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
            "protocol_sha256": file_sha(PROTOCOL_PATH), "teacher_provenance": teacher.provenance(),
            "exact_shared_leaf_contract": "one materialized contiguous 1024-token leaf tensor per split; offsets select tensor-identical leaves for every method",
            "datasets": {}, "artifacts": {}, "versions": {"torch": torch.__version__},
        }
        for key, spec in protocol["datasets"].items():
            manifest["datasets"][key] = {"id": spec["id"], "config": spec["config"], "revision": spec["revision"], "splits": {}}
            (stage / key).mkdir()
            for split in spec["splits"]:
                rows, fingerprint, features = load_rows(key, split, spec)
                path = stage / key / f"{split}.pt"
                stats = prepare_split(teacher, rows, path, leaf_batch=args.leaf_batch_size, direct_batch=args.direct_batch_size)
                stats["dataset_fingerprint"] = fingerprint
                stats["features"] = str(features)
                rel = str(path.relative_to(stage))
                manifest["datasets"][key]["splits"][split] = stats
                manifest["artifacts"][rel] = {"sha256": stats["artifact_sha256"], "bytes": stats["bytes"]}
        manifest["runtime_seconds"] = time.perf_counter() - overall
        mp = stage / "CACHE_MANIFEST.json"
        mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "COMPLETE").write_text(f"{file_sha(mp)}  CACHE_MANIFEST.json\n")
        os.replace(stage, OUTPUT)
        print(OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

if __name__ == "__main__":
    main()
