#!/usr/bin/env python3
"""Fail-closed V2 CLS TRAIN-only audit; does not load any protected source range."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/v2_cls")]
from protocol import TEACHER_ID, TEACHER_POOLING, TEACHER_REVISION, TRAIN_SCAN  # noqa: E402

CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")
SEEDS = (424242, 271828, 314159)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def audit_cache(cache):
    manifest = json.loads((cache / "CACHE_MANIFEST.json").read_text())
    if manifest.get("teacher_pooling") != "cls" or manifest.get("source_scan") != list(TRAIN_SCAN):
        raise ValueError("cache is not declared V2 CLS TRAIN")
    if manifest.get("teacher", {}).get("teacher_id") != TEACHER_ID or manifest["teacher"].get("teacher_revision") != TEACHER_REVISION:
        raise ValueError("cache teacher mismatch")
    per, total_rows, total_leaves, total_content_tokens = {}, 0, 0, 0
    for corpus in CORPORA:
        rows = torch.load(cache / corpus / "train.pt", map_location="cpu", weights_only=False)
        checked_tokens = 0
        for row in rows:
            if row.get("teacher_pooling") != "cls" or row.get("teacher_id") != TEACHER_ID or row.get("teacher_revision") != TEACHER_REVISION:
                raise ValueError(f"row provenance mismatch {corpus}:{row.get('id')}")
            lengths = row["lengths"]
            if not torch.isfinite(lengths).all() or (lengths <= 0).any():
                raise ValueError("invalid token masses")
            # Exact target token spans are serialized as contiguous leaf intervals.
            for target, p in zip(row["targets"], row["target_provenance"]):
                expected = int(lengths[int(p["leaf_start"]):int(p["leaf_end"])].sum().item())
                if expected != int(p["token_count"]) or bool(target is not None) != bool(p["teacher_eligible"]):
                    raise ValueError("target token accounting/provenance mismatch")
                checked_tokens += expected
            total_leaves += len(lengths); total_content_tokens += int(lengths.sum().item())
        per[corpus] = {"rows": len(rows), "target_span_tokens_checked": checked_tokens}
        total_rows += len(rows)
    return {"manifest_sha256": sha256(cache / "CACHE_MANIFEST.json"), "rows": total_rows,
            "leaves": total_leaves, "content_tokens": total_content_tokens, "per_corpus": per}


def audit_seed(seed):
    base = ROOT / f"results/v2_cls/final_seed{seed}"
    a_status = json.loads((base / "stage_a" / f"encoder_seed{seed}_run_status.json").read_text())
    item = {"seed": seed, "stage_a_status": a_status}
    if a_status["status"] != "completed":
        item["final_checkpoint"] = None
        item["ready"] = False
        return item
    for stage, variant in (("stage_b", "autoencoder"), ("stage_c", "autoencoder")):
        status_path = base / stage / f"{variant}_seed{seed}_run_status.json"
        status = json.loads(status_path.read_text())
        if status["status"] != "completed":
            raise RuntimeError(f"{seed} {stage} did not complete")
        item[f"{stage}_status"] = status
    checkpoint = base / "stage_c" / f"autoencoder_seed{seed}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {"teacher_id": TEACHER_ID, "teacher_revision": TEACHER_REVISION, "teacher_pooling": TEACHER_POOLING}
    if payload.get("teacher_provenance") != expected or payload.get("seed") != seed or payload.get("selected_step") != 1200:
        raise ValueError(f"final checkpoint identity failure seed={seed}")
    config = payload.get("config") or {}
    if config.get("v2_stage") != "stage_c" or config.get("continue_autoencoder_from_checkpoint") is None:
        raise ValueError("final checkpoint lacks V2 stage-C ancestry")
    if any("v1" in str(value).lower() for key, value in config.items() if "checkpoint" in key):
        raise ValueError("v1 initialization reference found")
    item["final_checkpoint"] = {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}
    item["ready"] = True
    return item


def main():
    cache = ROOT / "artifacts/v2_cls/cache_train_cls"
    seeds = [audit_seed(s) for s in SEEDS]
    timings = []
    for path in ROOT.glob("results/v2_cls/**/*_timing.json"):
        timings.append({"path": str(path.relative_to(ROOT)), **json.loads(path.read_text())})
    result = {
        "role": "train_only_audit", "development_accessed": False, "official_accessed": False,
        "teacher": {"teacher_id": TEACHER_ID, "teacher_revision": TEACHER_REVISION, "teacher_pooling": TEACHER_POOLING},
        "cache": audit_cache(cache), "seeds": seeds, "timings": sorted(timings, key=lambda x: x["path"]),
        "all_three_ready": all(x["ready"] for x in seeds),
    }
    out = ROOT / "results/v2_cls/FINAL_TRAIN_AUDIT.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
