"""Predeclared V2 CLS protocol guards.

This module intentionally does not import any corpus loader or open official data.
It records the fresh source-index ranges selected after all known v1 spent ranges.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (424242, 271828, 314159)
TEACHER_ID = "Alibaba-NLP/gte-modernbert-base"
TEACHER_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
TEACHER_POOLING = "cls"
CACHE_NAMESPACE = "v2_cls_kary4_train"
TRAIN_SCAN = (192000, 204000)
DEVELOPMENT_SCAN = (204000, 213000)
RESERVE_BUFFER = (213000, 214000)
FUTURE_OFFICIAL_SCAN = (214000, 244000)
SPENT_PRIOR_RANGES = {
    "official_v5": (110000, 140000),
    "clean_v6_train": (140000, 152000),
    "clean_v6_development": (152000, 161000),
    "clean_v6_buffer": (161000, 162000),
    "official_v6": (162000, 192000),
}
CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) >= min(a[1], b[1])


def assert_source_ranges_nonoverlap() -> dict:
    declared = {
        "v2_cls_train": TRAIN_SCAN,
        "v2_cls_development": DEVELOPMENT_SCAN,
        "v2_cls_buffer": RESERVE_BUFFER,
        "v2_cls_future_official": FUTURE_OFFICIAL_SCAN,
    }
    all_ranges = {**SPENT_PRIOR_RANGES, **declared}
    names = list(all_ranges)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if not disjoint(all_ranges[left], all_ranges[right]):
                raise ValueError(f"source ranges overlap: {left}={all_ranges[left]} {right}={all_ranges[right]}")
    if TRAIN_SCAN[1] != DEVELOPMENT_SCAN[0] or DEVELOPMENT_SCAN[1] != RESERVE_BUFFER[0] or RESERVE_BUFFER[1] != FUTURE_OFFICIAL_SCAN[0]:
        raise ValueError("V2 ranges must be contiguous with the declared buffer before official")
    return {k: list(v) for k, v in all_ranges.items()}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    validate_config(cfg, Path(path))
    return cfg


def validate_config(cfg: dict, path: Path | None = None) -> None:
    assert_source_ranges_nonoverlap()
    seed = int(cfg.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"unexpected V2 seed {seed}")
    required = {
        "model_name": TEACHER_ID,
        "model_revision": TEACHER_REVISION,
        "teacher_pooling": TEACHER_POOLING,
        "arity": 4,
        "cache_dir": "artifacts/v2_cls/cache_train_cls",
        "training_without_validation": True,
        "initialize_encoder_from_autoencoder_checkpoint": False,
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"V2 config {path or ''} has {key}={cfg.get(key)!r}, expected {value!r}")
    if cfg.get("initialize_from_encoder_checkpoint") or cfg.get("initialize_from_encoder_checkpoint_sha256"):
        raise ValueError("V2 CLS fresh protocol refuses learned v1 initialization checkpoints")
    if cfg.get("trust_remote_code") is not True:
        raise ValueError("V2 GTE ModernBERT config must explicitly set trust_remote_code: true")
    for corpus in cfg.get("corpora", []):
        interval = (int(corpus["train_start"]), int(corpus["train_start"]) + int(corpus["train_documents"]))
        if interval != TRAIN_SCAN:
            raise ValueError(f"{corpus['id']} train range {interval} != {TRAIN_SCAN}")


def audit_configs() -> dict:
    paths = sorted((ROOT / "configs/v2_cls").glob("seed*.yaml"))
    if len(paths) != len(SEEDS):
        raise ValueError("expected exactly three V2 seed configs")
    out = {"ranges": assert_source_ranges_nonoverlap(), "configs": {}}
    for path in paths:
        cfg = load_config(path)
        out["configs"][path.name] = {"seed": cfg["seed"], "sha256": sha256_file(path)}
    return out


if __name__ == "__main__":
    print(json.dumps(audit_configs(), indent=2) + "\n")
