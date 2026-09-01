"""Atomically prepare the fresh v6 TRAIN-only cache from [140000,152000)."""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from recursive_geodesic_composition.data import FrozenTeacher, stream_document_ranges  # noqa: E402
from recursive_geodesic_composition.prepare_kary import prepare_kary_split, require_free_storage  # noqa: E402
from protocol import (CORPORA, DEVELOPMENT_SCAN, DEV_CACHE, ROOT, TRAIN_CACHE,  # noqa: E402
                      TRAIN_SCAN, atomic_json, load_config, reject_path,
                      sha256_file, stable_hash)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def prepare(config_path: Path, device: str) -> Path:
    cfg = load_config(config_path)
    destination = ROOT / TRAIN_CACHE
    marker = destination.with_name(destination.name + ".PREPARATION_STARTED")
    reject_path(destination)
    if destination.exists() or marker.exists():
        raise FileExistsError("v6 TRAIN namespace is create-only")
    if (ROOT / DEV_CACHE).exists():
        raise RuntimeError("development cache must not precede fresh TRAIN preparation")
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"config_sha256={sha256_file(config_path)}\nauthorized_scan={TRAIN_SCAN}\n")
        handle.flush(); os.fsync(handle.fileno())
    require_free_storage(ROOT, 1800000000, "v6 TRAIN cache preparation")
    stage = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        teacher = FrozenTeacher(cfg["model_name"], cfg["model_revision"], device)
        stage_cfg = copy.deepcopy(cfg)
        stage_cfg["cache_dir"] = str(stage)
        corpus_audit = {}
        for corpus in cfg["corpora"]:
            documents = stream_document_ranges(
                corpus["dataset_name"], corpus.get("dataset_config"), corpus["dataset_revision"],
                {"train": (TRAIN_SCAN[0], TRAIN_SCAN[1] - TRAIN_SCAN[0])},
            )["train"]
            prepare_kary_split(stage_cfg, teacher, corpus, "train", documents)
            sidecar_path = stage / corpus["id"] / "train.pt.json"
            sidecar = json.loads(sidecar_path.read_text())
            if (sidecar.get("source_start"), sidecar.get("requested")) != (TRAIN_SCAN[0], TRAIN_SCAN[1] - TRAIN_SCAN[0]):
                raise ValueError("TRAIN sidecar escaped authorized source range")
            corpus_audit[corpus["id"]] = {
                "source_rows_scanned": TRAIN_SCAN[1] - TRAIN_SCAN[0],
                "usable_augmented_rows": int(sidecar["usable"]),
                "selection": "all_nonblank_source_rows_then_deterministic_document_seeded_augmentations",
            }
        artifacts = {}
        for path in sorted(stage.glob("*/*")):
            if path.is_file():
                artifacts[str(path.relative_to(stage))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "study": cfg["study"].rsplit("_seed", 1)[0],
            "cache_namespace": TRAIN_CACHE,
            "role": "train",
            "development_only": True,
            "protected_data_accessed": False,
            "atomic_create_only": True,
            "source_scan": list(TRAIN_SCAN),
            "next_development_scan": list(DEVELOPMENT_SCAN),
            "next_development_opened": False,
            "corpora": list(CORPORA),
            "corpus_pins_sha256": stable_hash([(c["id"], c["dataset_name"], c.get("dataset_config"), c["dataset_revision"]) for c in cfg["corpora"]]),
            "cache_construction_seed": int(cfg["seed"]),
            "deterministic_outcome_blind": True,
            "corpus_audit": corpus_audit,
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "preparer_sha256": sha256_file(__file__),
            "artifacts": artifacts,
        }
        atomic_json(stage / "CACHE_MANIFEST.json", manifest)
        (stage / "COMPLETE").write_text(f"{sha256_file(stage / 'CACHE_MANIFEST.json')}  CACHE_MANIFEST.json\n")
        _fsync_tree(stage)
        os.replace(stage, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/final_v6/seed424242.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(prepare(Path(args.config).resolve(), args.device))
