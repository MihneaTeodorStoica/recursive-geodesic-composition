#!/usr/bin/env python3
"""Validate V2 CLS extraction against documented GTE reference usage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from recursive_geodesic_composition.data import FrozenTeacher  # noqa: E402

TEXTS = [
    "Recursive geodesic composition merges ordered document chunks.",
    "The documented GTE ModernBERT embedding is the normalized CLS token.",
    "A short text.",
    "Longer biomedical retrieval passages should not use a masked mean by accident.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--revision", default="e7f32e3c00f91d699e8c43b53106206bcc72bb22")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="reproducibility/v2_cls/CLS_REFERENCE_VALIDATION.json")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=True).to(device).eval()
    encoded = tokenizer(TEXTS, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**encoded)
        reference = F.normalize(outputs.last_hidden_state[:, 0].float(), dim=-1).cpu()
    teacher = FrozenTeacher(args.model, args.revision, str(device), pooling="cls", trust_remote_code=True)
    token_chunks = [tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"] for t in TEXTS]
    wrapped = teacher.embed(token_chunks, batch_size=len(token_chunks))
    max_abs = float((reference - wrapped).abs().max().item())
    cos = F.cosine_similarity(reference, wrapped).tolist()
    result = {
        "model": args.model,
        "revision": args.revision,
        "pooling": "cls",
        "formula": "torch.nn.functional.normalize(outputs.last_hidden_state[:, 0].float(), dim=-1)",
        "texts": TEXTS,
        "max_abs_difference": max_abs,
        "cosines": cos,
        "passed": max_abs <= args.atol and min(cos) >= args.min_cosine,
        "teacher_provenance": teacher.provenance(),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("CLS reference validation failed")


if __name__ == "__main__":
    main()
