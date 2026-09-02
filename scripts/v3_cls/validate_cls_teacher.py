#!/usr/bin/env python
"""Numerically validate repository CLS extraction against direct model-card usage."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text  # noqa:E402

MODEL = "Alibaba-NLP/gte-modernbert-base"
REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
TEXTS = [
    "Recursive geodesic composition should preserve document meaning across reductions.",
    "A short control sentence.",
    "ModernBERT CLS pooling is normalized after selecting the first hidden state.",
]

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

@torch.inference_mode()
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='reproducibility/v3_cls/CLS_REFERENCE_VALIDATION.json')
    args=ap.parse_args()
    teacher=FrozenTeacher(MODEL, REVISION, args.device, pooling='cls', trust_remote_code=True)
    token_chunks=[tokenize_text(teacher.tokenizer, text) for text in TEXTS]
    repo=teacher.embed(token_chunks, batch_size=2)
    # Compare to direct documented usage with the same batching/padding shape as
    # `FrozenTeacher.embed`; this isolates pooling/tokenization identity rather
    # than ModernBERT's small pad-length-dependent numerical variation.
    direct_parts=[]
    for i in range(0, len(TEXTS), 2):
        encoded=teacher.tokenizer(TEXTS[i:i+2], padding=True, truncation=True,
                                  max_length=getattr(teacher.model.config, 'max_position_embeddings', 8192),
                                  return_tensors='pt').to(teacher.device)
        direct_parts.append(F.normalize(teacher.model(**encoded).last_hidden_state[:,0].float(), dim=-1).cpu())
    direct=torch.cat(direct_parts)
    diff=(repo-direct).abs()
    result={
        'teacher_provenance': teacher.provenance(),
        'texts_sha256': hashlib.sha256(json.dumps(TEXTS, sort_keys=True).encode()).hexdigest(),
        'max_abs_diff': float(diff.max().item()),
        'mean_abs_diff': float(diff.mean().item()),
        'cosine_min': float((repo*direct).sum(-1).min().item()),
        'passed': bool(diff.max().item() <= 1e-6 and (repo*direct).sum(-1).min().item() >= 1-1e-6),
        'implementation_hashes': {
            'data_py_sha256': sha256_file(ROOT/'src/recursive_geodesic_composition/data.py'),
            'validation_script_sha256': sha256_file(__file__),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
