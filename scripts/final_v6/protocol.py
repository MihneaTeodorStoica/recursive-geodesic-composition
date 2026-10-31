"""Fail-closed CLEAN-DEVELOPMENT-only protocol for strict-K4 AE v6."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = "kary4_strict_autoencoder_v6_clean_development"
TRAIN_CACHE = "cache_kary4_strict_autoencoder_v6_clean_train"
DEV_CACHE = "cache_kary4_strict_autoencoder_v6_clean_development"
AGGREGATE_OUTPUT = "results_kary4_strict_autoencoder_v6_clean_development"
CORPORA = ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101")
SEEDS = (424242, 271828, 314159)
TRAIN_SCAN = (140000, 152000)
DEVELOPMENT_SCAN = (152000, 161000)
RESERVE_BUFFER = (161000, 162000)
FUTURE_PROTECTED = (162000, 192000)
SAFE_RANGE = (1025, 4096)
TARGET_DOCUMENTS = 1000
FORBIDDEN_PATH_TERMS = ("official", "test")
PINS = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc"),
    "openwebtext": ("Skylion007/openwebtext", None, "79d93d786212f7344586290adb811d4ae6a1762c"),
    "wikipedia_en_20231101": ("wikimedia/wikipedia", "20231101.en", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"),
}
SOURCE_CHECKPOINTS = {
    424242: ("results_kary4_strict_autoencoder_seed424_dev_v2/autoencoder_seed424242.pt", "2365a789f61ba6dab9967041b8dcee2a19dc969fd97b71026bdee970f7ea4797"),
    271828: ("results_kary4_strict_autoencoder_multiseed_confirmation_v1_seed271828/autoencoder_seed271828.pt", "63103ce0b940dab6eec02c62ae5e177b0f23941f1775624b2fe3b27f60105603"),
    314159: ("results_kary4_strict_autoencoder_multiseed_confirmation_v1_seed314159/autoencoder_seed314159.pt", "27116bb5dbe279750264147100b38323ec432329b385404f59db5bf753d4b3a2"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reject_path(path: str | Path) -> None:
    if any(term in str(path).lower() for term in FORBIDDEN_PATH_TERMS):
        raise ValueError(f"protected-role term in v6 artifact path: {path}")


def disjoint(left, right) -> bool:
    return int(left[1]) <= int(right[0]) or int(right[1]) <= int(left[0])


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_config(cfg: dict) -> None:
    seed = int(cfg.get("seed", -1))
    if seed not in SEEDS:
        raise ValueError("undeclared v6 seed")
    expected_source = SOURCE_CHECKPOINTS[seed]
    exact = {
        "study": f"{STUDY}_seed{seed}", "development_only": True,
        "protected_data_accessed": False, "arity": 4, "cache_dir": TRAIN_CACHE,
        "output_dir": f"results_{STUDY}_seed{seed}", "training_without_validation": True,
        "fixed_final_checkpoint_without_validation": True,
        "initialize_encoder_from_autoencoder_checkpoint": True,
        "initialize_from_encoder_checkpoint": expected_source[0],
        "initialize_from_encoder_checkpoint_sha256": expected_source[1],
        "batch_size": 8, "steps": 400, "validation_every": 400,
        "freeze_encoder_steps": 300, "decoder_only_require_full_arity_rows": True,
        "autoencoder_objective_start_step": 300, "autoencoder_objective_ramp_steps": 100,
        "reconstruction_weight": .15, "cycle_weight": 5.0,
        "importance_loss_weight": .025, "distribution_mean_weight": .25,
        "distribution_variance_weight": 256.0, "learning_rate": 2e-5,
        "encoder_learning_rate_scale": .02, "learning_rate_schedule": "cosine",
        "warmup_steps": 50, "min_learning_rate": 2e-6,
        "decoder_mode": "strict", "decoder_length_mode": "predicted",
        "importance_mode": "semantic", "geodesic_parameterization": "tangent_expmap",
        "stratified_multicorpus_batches": True,
    }
    for key, value in exact.items():
        if cfg.get(key) != value:
            raise ValueError(f"v6 recipe/config changed: {key}")
    if tuple(item.get("id") for item in cfg.get("corpora", ())) != CORPORA:
        raise ValueError("corpus order changed")
    for item in cfg["corpora"]:
        if (item.get("dataset_name"), item.get("dataset_config"), item.get("dataset_revision")) != PINS[item["id"]]:
            raise ValueError(f"corpus pin changed: {item['id']}")
        if (int(item["train_start"]), int(item["train_documents"])) != (TRAIN_SCAN[0], TRAIN_SCAN[1] - TRAIN_SCAN[0]):
            raise ValueError("fresh TRAIN scan changed")
        if float(item["weight"]) != .25:
            raise ValueError("equal corpus weight changed")
    for key in ("cache_dir", "output_dir", "initialize_from_encoder_checkpoint", "attack_config"):
        reject_path(cfg[key])
    ranges = (TRAIN_SCAN, DEVELOPMENT_SCAN, RESERVE_BUFFER, FUTURE_PROTECTED)
    if any(not disjoint(a, b) for i, a in enumerate(ranges) for b in ranges[i + 1:]):
        raise ValueError("v6 ranges overlap")
    if TRAIN_SCAN[1] != DEVELOPMENT_SCAN[0] or DEVELOPMENT_SCAN[1] != RESERVE_BUFFER[0] or RESERVE_BUFFER[1] != FUTURE_PROTECTED[0]:
        raise ValueError("v6 range adjacency changed")


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    validate_config(cfg)
    return cfg
