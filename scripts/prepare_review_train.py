#!/usr/bin/env python3
"""New TRAIN-only exact-four prefixes; never replace the historical TRAIN cache."""
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from recursive_geodesic_composition.data import tokenize_text
from prepare_safe_baselines import token_hash
from review_common import *


def main():
    root, cfg, device, cal, baseline, official, published = begin("review_train", __file__, cache=True)
    section = cfg["train"]
    tokenizer = AutoTokenizer.from_pretrained(official["model_name"], revision=official["model_revision"])
    evaluation = read_tokens(baseline, published)
    excluded = {r["safe_prefix_token_ids_sha256"] for r in evaluation}
    seen, rows, audit = set(excluded), [], {}
    for corpus in official["corpora"]:
        name = corpus["id"]
        data = load_dataset(corpus["dataset_name"], corpus.get("dataset_config"), split="train", streaming=True, revision=corpus["dataset_revision"])
        chosen, duplicates, scanned = [], 0, 0
        start, stop = section["source_start"], section["source_end_exclusive"]
        for offset, raw in enumerate(data.skip(start).take(stop-start)):
            scanned += 1
            text = raw.get("text") or raw.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = tokenize_text(tokenizer, text)[:4096]
            if len(tokens) <= 3072:
                continue
            digest = token_hash(tokens)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            rank = len(chosen)
            chosen.append({"id": f"{name}:{start+offset}", "corpus": name, "source_index": start+offset,
                "role": "fit" if rank < section["fit_per_corpus"] else "validation", "tokens": tokens, "token_sha256": digest})
            if len(chosen) == section["fit_per_corpus"] + section["validation_per_corpus"]:
                break
        if len(chosen) != section["fit_per_corpus"] + section["validation_per_corpus"]:
            raise RuntimeError(f"insufficient fixed-range TRAIN documents: {name}")
        rows.extend(chosen)
        audit[name] = {"scanned": scanned, "duplicates_excluded": duplicates, "selected": len(chosen), "ids": [r["id"] for r in chosen]}
        print(f"Selected {name}: {len(chosen)} in {scanned} source rows", flush=True)
    atomic_tensor(root / "tokens.pt", rows)
    teacher = make_teacher(official, device)
    output = []
    for i, row in enumerate(rows):
        tokens = row["tokens"]
        chunks = [tokens[j:j+1024] for j in range(0, len(tokens), 1024)]
        leaves = teacher.embed(chunks, 4)
        target = teacher.embed([tokens], 1)[0]
        validate_vectors(leaves.numpy(), unit=True)
        validate_vectors(target[None].numpy(), unit=True)
        output.append({k: v for k, v in row.items() if k != "tokens"} | {"leaves": leaves, "lengths": torch.tensor([len(c) for c in chunks], dtype=torch.float32), "teacher_target": target})
        if (i+1) % 100 == 0:
            print(f"TRAIN encoded {i+1}/{len(rows)}", flush=True)
    atomic_tensor(root / "rows.pt", output)
    seal(root, {"documents": len(rows), "fit": sum(r["role"] == "fit" for r in rows), "validation": sum(r["role"] == "validation" for r in rows),
                "selection": audit, "rows_sha256": sha256(root / "rows.pt"), "tokens_sha256": sha256(root / "tokens.pt"),
                "evaluation_token_overlap": len(excluded & {r["token_sha256"] for r in rows}), "original_cache_identity_claimed": False})


if __name__ == "__main__":
    main()
