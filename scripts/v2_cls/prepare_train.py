#!/usr/bin/env python3
"""Create-only V2 CLS TRAIN cache after protocol/range validation."""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from recursive_geodesic_composition.data import FrozenTeacher, stream_document_ranges  # noqa: E402
from recursive_geodesic_composition.prepare_kary import prepare_kary_split, require_free_storage  # noqa: E402
from protocol import CACHE_NAMESPACE, CORPORA, DEVELOPMENT_SCAN, TEACHER_POOLING, TRAIN_SCAN, load_config, sha256_file  # noqa: E402


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def prepare(config_path: Path, device: str) -> Path:
    cfg = load_config(config_path)
    destination = ROOT / cfg["cache_dir"]
    marker = destination.with_name(destination.name + ".PREPARATION_STARTED")
    if destination.exists() or marker.exists():
        raise FileExistsError(f"V2 CLS TRAIN namespace is create-only: {destination}")
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as handle:
        handle.write(f"config_sha256={sha256_file(config_path)}\nauthorized_scan={TRAIN_SCAN}\nteacher_pooling={TEACHER_POOLING}\n")
        handle.flush(); os.fsync(handle.fileno())
    require_free_storage(ROOT, 1800000000, "V2 CLS TRAIN cache preparation")
    stage = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir(parents=True)
    try:
        teacher = FrozenTeacher(
            cfg["model_name"], cfg["model_revision"], device,
            pooling=cfg["teacher_pooling"], trust_remote_code=cfg.get("trust_remote_code"))
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
            if sidecar.get("teacher_pooling") != "cls":
                raise ValueError("V2 cache sidecar did not record CLS teacher pooling")
            corpus_audit[corpus["id"]] = {
                "source_rows_scanned": TRAIN_SCAN[1] - TRAIN_SCAN[0],
                "usable_augmented_rows": int(sidecar["usable"]),
                "sidecar_sha256": sha256_file(sidecar_path),
            }
        artifacts = {}
        for path in sorted(stage.glob("*/*")):
            if path.is_file():
                artifacts[str(path.relative_to(stage))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "cache_namespace": CACHE_NAMESPACE,
            "role": "train",
            "teacher": teacher.provenance(),
            "teacher_pooling": "cls",
            "source_scan": list(TRAIN_SCAN),
            "next_development_scan": list(DEVELOPMENT_SCAN),
            "prior_ranges_refused": True,
            "corpora": list(CORPORA),
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "preparer_sha256": sha256_file(Path(__file__)),
            "corpus_audit": corpus_audit,
            "artifacts": artifacts,
        }
        (stage / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "COMPLETE").write_text(f"{sha256_file(stage / 'CACHE_MANIFEST.json')}  CACHE_MANIFEST.json\n")
        _fsync_tree(stage)
        os.replace(stage, destination)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/v2_cls/seed424242.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(prepare(Path(args.config).resolve(), args.device))
