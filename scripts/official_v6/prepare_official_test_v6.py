"""Atomic one-shot preparation of both official-v6 arms on identical documents.

Importing this module cannot load a dataset. ``prepare`` is the sole data entry
point, and the lazy stream authorizes only the frozen source window. Every prior
index, including clean-v6 TRAIN/development and the buffer, is refused.
The full arm embeds every token and fails rather than imposing an operational cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import torch

from official_test_v6_protocol import (
    ARITY,
    CACHE_NAMESPACE,
    CHUNK_TOKENS,
    CONFIG,
    CORPUS_ORDER,
    PRIOR_RANGES,
    ROLE,
    ROOT,
    SAFE_RANGE,
    SCAN,
    TARGET_DOCUMENTS,
    atomic_text,
    atomic_torch_save,
    claim_once,
    exact_chunks,
    fsync_directory,
    load_config,
    reduction_depth,
    require_storage,
    safe_prefix_length,
    sha256_file,
    stable_hash,
    validate_immutable_provenance,
)


def _stream_scan_window(corpus: dict, start: int, end_exclusive: int):
    """Lazily yield only the exact authorized untouched source window."""
    if (int(start), int(end_exclusive)) != SCAN:
        raise PermissionError(
            f"refusing every prior/outside range; only {SCAN} is authorized"
        )
    from datasets import (
        load_dataset,
    )  # Lazy: imports/tests cannot access official data.

    dataset = load_dataset(
        corpus["dataset_name"],
        corpus.get("dataset_config"),
        split="train",
        streaming=True,
        revision=corpus["dataset_revision"],
    )
    maximum = end_exclusive - start
    for offset, raw in enumerate(dataset.skip(start).take(maximum)):
        if offset >= maximum:
            raise AssertionError("dataset stream escaped authorized v6 scan")
        yield start + offset, raw


def _token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


@torch.inference_mode()
def embed_every_leaf(
    teacher, token_ids: list[int], batch_size: int
) -> tuple[torch.Tensor, list[int], float]:
    """Embed all exact contiguous leaves in bounded operational batches.

    Batch size limits memory only. There is deliberately no token/leaf/document
    limit and no exception is converted into truncation or document skipping.
    """
    chunks = exact_chunks(token_ids)
    outputs = []
    started = time.perf_counter()
    for start in range(0, len(chunks), int(batch_size)):
        part = chunks[start : start + int(batch_size)]
        embedded = teacher.embed(part, int(batch_size)).float()
        if embedded.shape[0] != len(part) or not torch.isfinite(embedded).all():
            raise FloatingPointError(
                "full-document leaf embedder omitted or corrupted a leaf"
            )
        outputs.append(embedded)
    elapsed = time.perf_counter() - started
    leaves = torch.cat(outputs) if outputs else torch.empty((0, 0))
    lengths = [len(chunk) for chunk in chunks]
    if (
        not lengths
        or sum(lengths) != len(token_ids)
        or len(lengths) != (len(token_ids) + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    ):
        raise AssertionError("full-document token/leaf accounting mismatch")
    return leaves, lengths, elapsed


def validate_safe_row(row: dict, corpus: str, actual_end: int) -> None:
    if row.get("cache_namespace") != CACHE_NAMESPACE or row.get("cache_role") != ROLE:
        raise ValueError("safe row namespace/role mismatch")
    source_index = int(row.get("source_index", -1))
    if not SCAN[0] <= source_index < min(SCAN[1], int(actual_end)):
        raise ValueError("safe row source escaped v6 scan")
    if row.get("id") != f"{corpus}:{source_index}" or row.get("corpus") != corpus:
        raise ValueError("safe row group/corpus mismatch")
    if (
        row.get("arm") != "safe_teacher_comparable"
        or row.get("reduction") != "direct_single_k4"
    ):
        raise ValueError("safe row arm/reduction mismatch")
    if (
        int(row.get("arity", -1)) != ARITY
        or int(row.get("teacher_max_tokens", -1)) != SAFE_RANGE[1]
    ):
        raise ValueError("safe row K4/teacher cap mismatch")
    leaves, lengths, target = (
        row.get("leaves"),
        row.get("lengths"),
        row.get("teacher_target"),
    )
    if (
        not isinstance(leaves, torch.Tensor)
        or not isinstance(lengths, torch.Tensor)
        or not isinstance(target, torch.Tensor)
    ):
        raise TypeError("safe leaves/lengths/teacher target must be tensors")
    if (
        leaves.ndim != 2
        or lengths.ndim != 1
        or target.ndim != 1
        or len(leaves) != len(lengths)
        or leaves.shape[1] != 768
        or target.shape[0] != 768
    ):
        raise ValueError("safe tensor ranks/counts/dimension are malformed")
    values = [int(value) for value in lengths.tolist()]
    covered = int(row.get("safe_prefix_token_count", -1))
    if (
        len(values) not in (2, 3, 4)
        or sum(values) != covered
        or not SAFE_RANGE[0] <= covered <= SAFE_RANGE[1]
    ):
        raise ValueError("safe prefix does not cover exactly 1025..4096 tokens")
    if any(value <= 0 or value > CHUNK_TOKENS for value in values) or any(
        value != CHUNK_TOKENS for value in values[:-1]
    ):
        raise ValueError("safe prefix chunks are not exact contiguous <=1024 leaves")
    if not torch.isfinite(leaves).all() or not torch.isfinite(target).all():
        raise FloatingPointError("safe row contains nonfinite leaf/teacher embedding")
    if int(row.get("full_token_count", -1)) < covered:
        raise ValueError("full token count is smaller than safe prefix")


def validate_full_shard(shard: dict, expected: dict) -> None:
    forbidden = {
        "teacher_target",
        "targets",
        "full_root_teacher_target",
        "teacher_cosine",
        "auc",
    }
    if forbidden & set(shard):
        raise ValueError("full shard contains a forbidden teacher/comparison field")
    fixed = {
        "cache_namespace": CACHE_NAMESPACE,
        "cache_role": ROLE,
        "arm": "unrestricted_full_document_recursion",
        "arity": ARITY,
        "chunk_tokens": CHUNK_TOKENS,
        "reduction": "deterministic_contiguous_k4_carry_singleton",
        "truncated": False,
        "token_cap": None,
        "maximum_leaves": None,
        "recursion_depth_cap": None,
        "leaf_storage_dtype": "float16",
    }
    for key, value in fixed.items():
        if shard.get(key) != value:
            raise ValueError(f"full shard invariant mismatch: {key}")
    for key in (
        "id",
        "source_index",
        "source_document_id",
        "corpus",
        "full_token_count",
        "full_leaf_count",
        "tree_depth",
    ):
        if shard.get(key) != expected.get(key):
            raise ValueError(f"full shard/index mismatch: {key}")
    leaves, lengths = shard.get("leaves"), shard.get("lengths")
    if (
        not isinstance(leaves, torch.Tensor)
        or leaves.dtype != torch.float16
        or leaves.ndim != 2
        or leaves.shape[1] != 768
    ):
        raise TypeError("full leaves must be rank-2 fp16 with dimension 768")
    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.ndim != 1
        or len(lengths) != len(leaves)
    ):
        raise TypeError("full lengths must align one-for-one with leaves")
    values = [int(value) for value in lengths.tolist()]
    if len(values) != int(shard["full_leaf_count"]) or sum(values) != int(
        shard["full_token_count"]
    ):
        raise ValueError("full shard exact token/leaf accounting mismatch")
    if any(value <= 0 or value > CHUNK_TOKENS for value in values) or any(
        value != CHUNK_TOKENS for value in values[:-1]
    ):
        raise ValueError("full shard contains noncontiguous/nonexact chunk lengths")
    if reduction_depth(len(values)) != int(shard["tree_depth"]):
        raise ValueError("full shard depth accounting mismatch")
    if not torch.isfinite(leaves).all():
        raise FloatingPointError("full shard contains nonfinite leaves")


def _distribution(values: list[int]) -> dict:
    ordered = sorted(map(int, values))
    if not ordered:
        raise ValueError("cannot summarize empty values")

    def percentile(p: float) -> float:
        position = p * (len(ordered) - 1)
        low, high = int(position), min(int(position) + 1, len(ordered) - 1)
        fraction = position - low
        return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)

    return {
        "count": len(ordered),
        "sum": int(sum(ordered)),
        "minimum": int(ordered[0]),
        "mean": float(sum(ordered) / len(ordered)),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "maximum": int(ordered[-1]),
    }


def prepare(config_path: Path = CONFIG, device_name: str = "cuda") -> Path:
    cfg = load_config(config_path)
    # Mandatory before marker/cache/dataset access.
    immutable = validate_immutable_provenance(cfg, check_checkpoint_tensors=True)
    cache = ROOT / cfg["cache_dir"]
    marker = cache.with_name(cache.name + ".PREPARATION_STARTED")
    staging = cache.with_name(cache.name + f".staging.{uuid.uuid4().hex}")
    claim_once(marker, (cache,))
    staging.mkdir(parents=True)
    try:
        from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text

        teacher = FrozenTeacher(cfg["model_name"], cfg["model_revision"], device_name)
        context = int(getattr(teacher.model.config, "max_position_embeddings", 0)) - 2
        if context < SAFE_RANGE[1]:
            raise ValueError(
                f"frozen teacher context {context} cannot safely embed declared 4096-token targets"
            )
        if int(getattr(teacher.model.config, "hidden_size", -1)) != int(
            cfg["embedding_dimension"]
        ):
            raise ValueError("frozen teacher embedding dimension changed")

        manifest_artifacts, corpus_summaries, global_ids = {}, {}, set()
        reserve = int(cfg["storage"]["minimum_free_bytes_reserve"])
        overhead = int(cfg["storage"]["per_shard_overhead_bytes"])
        for corpus_cfg in cfg["corpora"]:
            corpus = str(corpus_cfg["id"])
            if corpus not in CORPUS_ORDER:
                raise ValueError("undeclared corpus")
            safe_rows, index_rows, source_identities = [], [], set()
            counters = {
                "scanned": 0,
                "accepted": 0,
                "excluded_blank_or_nontext": 0,
                "excluded_safe_prefix_too_short": 0,
            }
            leaf_embed_seconds = target_embed_seconds = 0.0
            last_scanned = None
            for source_index, raw in _stream_scan_window(corpus_cfg, *SCAN):
                if not SCAN[0] <= source_index < SCAN[1]:
                    raise AssertionError("source stream escaped exact v6 window")
                counters["scanned"] += 1
                last_scanned = source_index
                text = raw.get("text") or raw.get("content") or ""
                if not isinstance(text, str) or not text.strip():
                    counters["excluded_blank_or_nontext"] += 1
                    continue
                token_ids = tokenize_text(teacher.tokenizer, text)
                prefix_count = safe_prefix_length(len(token_ids))
                if prefix_count is None:
                    counters["excluded_safe_prefix_too_short"] += 1
                    continue
                # Eligibility is now final. Every later operational error is fatal;
                # no document may be skipped or shortened.
                row_id = f"{corpus}:{source_index}"
                raw_source_id = str(raw.get("id", source_index))
                source_identity = f"{corpus}:{raw_source_id}"
                if row_id in global_ids or source_identity in source_identities:
                    raise ValueError(
                        f"duplicate accepted source group: {row_id}/{source_identity}"
                    )
                full_count = len(token_ids)
                full_leaf_count = (full_count + CHUNK_TOKENS - 1) // CHUNK_TOKENS
                estimated = (
                    full_leaf_count * int(cfg["embedding_dimension"]) * 2 + overhead
                )
                require_storage(
                    staging, reserve + estimated, f"unrestricted full shard {row_id}"
                )
                full_leaves, full_lengths, embed_seconds = embed_every_leaf(
                    teacher, token_ids, int(cfg["batches"]["leaf_embedding"])
                )
                leaf_embed_seconds += embed_seconds
                if tuple(full_leaves.shape) != (
                    full_leaf_count,
                    int(cfg["embedding_dimension"]),
                ):
                    raise ValueError("full leaf embedding shape mismatch")
                safe_ids = token_ids[:prefix_count]
                safe_leaf_count = (prefix_count + CHUNK_TOKENS - 1) // CHUNK_TOKENS
                safe_lengths = full_lengths[:safe_leaf_count]
                if sum(safe_lengths) != prefix_count:
                    raise AssertionError(
                        "safe prefix is not an exact prefix of full leaves"
                    )
                target_started = time.perf_counter()
                target = teacher.embed(
                    [safe_ids], int(cfg["batches"]["teacher_target"])
                )[0].float()
                target_embed_seconds += time.perf_counter() - target_started
                if not torch.isfinite(target).all():
                    raise FloatingPointError(f"nonfinite frozen safe target: {row_id}")
                selected_rank = len(safe_rows)
                shard_relative = f"{corpus}/full/{selected_rank:06d}.pt"
                shard_path = staging / shard_relative
                metadata = {
                    "id": row_id,
                    "source_index": int(source_index),
                    "source_document_id": raw_source_id,
                    "corpus": corpus,
                    "full_token_count": full_count,
                    "full_leaf_count": full_leaf_count,
                    "tree_depth": reduction_depth(full_leaf_count),
                }
                shard = {
                    **metadata,
                    "cache_namespace": CACHE_NAMESPACE,
                    "cache_role": ROLE,
                    "arm": "unrestricted_full_document_recursion",
                    "arity": ARITY,
                    "chunk_tokens": CHUNK_TOKENS,
                    "reduction": "deterministic_contiguous_k4_carry_singleton",
                    "truncated": False,
                    "token_cap": None,
                    "maximum_leaves": None,
                    "recursion_depth_cap": None,
                    "leaf_storage_dtype": "float16",
                    "full_token_ids_sha256": _token_hash(token_ids),
                    "lengths": torch.tensor(full_lengths, dtype=torch.int32),
                    "leaves": full_leaves.to(torch.float16),
                }
                validate_full_shard(shard, metadata)
                atomic_torch_save(shard, shard_path)
                shard_hash = sha256_file(shard_path)
                index_rows.append(
                    {
                        **metadata,
                        "selected_rank": selected_rank,
                        "safe_prefix_token_count": prefix_count,
                        "safe_leaf_count": safe_leaf_count,
                        "full_token_ids_sha256": shard["full_token_ids_sha256"],
                        "shard": shard_relative,
                        "shard_sha256": shard_hash,
                        "shard_bytes": shard_path.stat().st_size,
                    }
                )
                safe_row = {
                    "id": row_id,
                    "source_index": int(source_index),
                    "selected_rank": selected_rank,
                    "source_document_id": raw_source_id,
                    "corpus": corpus,
                    "cache_namespace": CACHE_NAMESPACE,
                    "cache_role": ROLE,
                    "arm": "safe_teacher_comparable",
                    "arity": ARITY,
                    "reduction": "direct_single_k4",
                    "chunk_tokens": CHUNK_TOKENS,
                    "teacher_max_tokens": SAFE_RANGE[1],
                    "safe_prefix_token_count": prefix_count,
                    "full_token_count": full_count,
                    "safe_prefix_token_ids_sha256": _token_hash(safe_ids),
                    "full_token_ids_sha256": shard["full_token_ids_sha256"],
                    "leaves": full_leaves[:safe_leaf_count].clone(),
                    "lengths": torch.tensor(safe_lengths, dtype=torch.float32),
                    "teacher_target": target,
                }
                validate_safe_row(safe_row, corpus, source_index + 1)
                safe_rows.append(safe_row)
                global_ids.add(row_id)
                source_identities.add(source_identity)
                counters["accepted"] += 1
                if len(safe_rows) == TARGET_DOCUMENTS:
                    break
            if len(safe_rows) != TARGET_DOCUMENTS:
                raise RuntimeError(
                    f"{corpus}: found only {len(safe_rows)} eligible documents in {SCAN}; "
                    "refusing to extend range or relax eligibility"
                )
            actual_end = int(last_scanned) + 1
            if (
                not SCAN[0] < actual_end <= SCAN[1]
                or counters["scanned"] != actual_end - SCAN[0]
            ):
                raise AssertionError("actual source scan accounting mismatch")
            if counters["scanned"] != sum(
                counters[key]
                for key in (
                    "accepted",
                    "excluded_blank_or_nontext",
                    "excluded_safe_prefix_too_short",
                )
            ):
                raise AssertionError("selection accounting mismatch")
            for safe_row, indexed in zip(safe_rows, index_rows):
                validate_safe_row(safe_row, corpus, actual_end)
                if (
                    safe_row["id"] != indexed["id"]
                    or safe_row["source_index"] != indexed["source_index"]
                ):
                    raise AssertionError(
                        "safe/full arms do not use identical accepted documents"
                    )
            safe_path = staging / corpus / "safe.pt"
            index_path = staging / corpus / "full_index.json"
            atomic_torch_save(safe_rows, safe_path)
            atomic_text(
                index_path,
                json.dumps(
                    {
                        "cache_namespace": CACHE_NAMESPACE,
                        "cache_role": ROLE,
                        "corpus": corpus,
                        "documents": index_rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            token_counts = [row["full_token_count"] for row in index_rows]
            leaf_counts = [row["full_leaf_count"] for row in index_rows]
            depths = [row["tree_depth"] for row in index_rows]
            safe_counts = [row["safe_prefix_token_count"] for row in index_rows]
            elapsed = leaf_embed_seconds + target_embed_seconds
            summary = {
                **counters,
                "declared_scan_range": list(SCAN),
                "actual_scanned_range": [SCAN[0], actual_end],
                "selection_rule": "first_1000_safe_prefix_length_eligible_only",
                "safe_and_full_document_order_sha256": stable_hash(
                    [row["id"] for row in index_rows]
                ),
                "source_index_order_sha256": stable_hash(
                    [row["source_index"] for row in index_rows]
                ),
                "safe_prefix_tokens": _distribution(safe_counts),
                "full_tokens": _distribution(token_counts),
                "full_leaves": _distribution(leaf_counts),
                "full_depth": _distribution(depths),
                "preparation_throughput": {
                    "leaf_embedding_seconds": leaf_embed_seconds,
                    "safe_target_embedding_seconds": target_embed_seconds,
                    "total_embedding_seconds": elapsed,
                    "full_tokens_per_second": sum(token_counts)
                    / max(leaf_embed_seconds, 1e-12),
                    "full_leaves_per_second": sum(leaf_counts)
                    / max(leaf_embed_seconds, 1e-12),
                    "documents_per_second": len(index_rows) / max(elapsed, 1e-12),
                },
                "safe_tensor": str(safe_path.relative_to(staging)),
                "safe_tensor_sha256": sha256_file(safe_path),
                "full_index": str(index_path.relative_to(staging)),
                "full_index_sha256": sha256_file(index_path),
                "full_shard_count": len(index_rows),
                "full_shard_bytes": int(sum(row["shard_bytes"] for row in index_rows)),
            }
            for value in summary["preparation_throughput"].values():
                if not torch.isfinite(torch.tensor(value)) or value < 0:
                    raise FloatingPointError("nonfinite preparation throughput")
            corpus_summaries[corpus] = summary
            manifest_artifacts[summary["safe_tensor"]] = summary["safe_tensor_sha256"]
            manifest_artifacts[summary["full_index"]] = summary["full_index_sha256"]

        manifest = {
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
            "config_path": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256_file(CONFIG),
            "immutable_provenance": immutable,
            "corpora": corpus_summaries,
            "artifacts": manifest_artifacts,
        }
        manifest_path = staging / "CACHE_MANIFEST.json"
        atomic_text(
            manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest_hash = sha256_file(manifest_path)
        atomic_text(staging / "COMPLETE", f"{manifest_hash}  CACHE_MANIFEST.json\n")
        fsync_directory(staging)
        os.replace(staging, cache)
        fsync_directory(cache.parent)
        return cache
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prepare(args.config, args.device)


if __name__ == "__main__":
    main()
