#!/usr/bin/env python3
"""Create-only binding of a pre-frozen V2 stage template to a V2 checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/v2_cls")]
from protocol import TEACHER_ID, TEACHER_POOLING, TEACHER_REVISION  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    template, source, out = Path(args.template), Path(args.source), Path(args.out)
    if out.exists():
        raise FileExistsError(f"create-only bound config already exists: {out}")
    cfg = yaml.safe_load(template.read_text())
    expected_variant = cfg.pop("v2_expected_source_variant")
    expected_stage = cfg.pop("v2_expected_source_stage")
    expected_protocol_version = int(cfg.pop("v2_expected_source_protocol_version"))
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != expected_variant:
        raise ValueError(f"source variant mismatch: {checkpoint.get('variant')} != {expected_variant}")
    if checkpoint.get("teacher_provenance") != {"teacher_id": TEACHER_ID, "teacher_revision": TEACHER_REVISION, "teacher_pooling": TEACHER_POOLING}:
        raise ValueError("source checkpoint is not explicit V2 CLS provenance")
    source_cfg = checkpoint.get("config") or {}
    if source_cfg.get("v2_stage") != expected_stage or source_cfg.get("fresh_protocol_version") != expected_protocol_version:
        raise ValueError("source does not have expected frozen V2 stage identity")
    source_hash = sha256(source)
    if expected_variant == "encoder":
        cfg["initialize_encoder_from_autoencoder_checkpoint"] = False
        cfg["initialize_from_encoder_checkpoint"] = str(source)
        cfg["initialize_from_encoder_checkpoint_sha256"] = source_hash
    else:
        cfg["continue_autoencoder_from_checkpoint"] = str(source)
        cfg["continue_autoencoder_from_checkpoint_sha256"] = source_hash
    cfg["v2_bound_source"] = {"path": str(source), "sha256": source_hash, "variant": expected_variant, "stage": expected_stage}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps({"out": str(out), "out_sha256": sha256(out), "source_sha256": source_hash}, indent=2))

if __name__ == "__main__":
    main()
