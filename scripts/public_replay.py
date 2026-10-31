"""Exact public-release inputs for fresh anonymous cached replays.

Historical identities are evidence, not rewritten aliases of this release.
Package checksums establish agreement with the checkout, not authenticity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPLAY_ID = "anonymous_cached_replay_v1"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _inside(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"unsafe replay input path: {relative}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"replay input escapes checkout: {relative}")
    return path


def package_file_hash(root: Path, relative: str) -> str:
    """Check a retained input against the current package's exact identity."""
    root = Path(root)
    manifest = json.loads((root / "reproducibility/PACKAGE_MANIFEST.json").read_text())
    entries = [row for row in manifest["files"] if row["path"] == relative]
    if len(entries) != 1:
        raise ValueError(f"replay input is not uniquely packaged: {relative}")
    entry = entries[0]
    path = _inside(root, relative)
    digest = _sha256(path)
    if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
        raise ValueError(f"packaged replay input changed: {relative}")
    return digest


def public_cache_manifest(root: Path, directory: str) -> tuple[dict, dict]:
    """Verify every extracted public-cache file before accepting its metadata."""
    root = Path(root)
    inventory_path = "reproducibility/HOSTED_CACHES.json"
    inventory_digest = package_file_hash(root, inventory_path)
    release = json.loads((root / inventory_path).read_text())
    bundles = [row for row in release["inventory"]["archives"]
               if row["cache_directory"] == directory]
    if len(bundles) != 1:
        raise ValueError(f"cache is not uniquely listed in public release: {directory}")
    bundle = bundles[0]
    prefix = directory + "/"
    expected = {}
    for entry in bundle["files"]:
        if not entry["path"].startswith(prefix):
            raise ValueError("cache inventory contains a foreign directory")
        relative = entry["path"][len(prefix):]
        if relative in expected:
            raise ValueError("duplicate public cache inventory entry")
        expected[relative] = entry
    cache = _inside(root, directory)
    paths = list(cache.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("public cache contains a symbolic link")
    actual = {str(path.relative_to(cache)) for path in paths if path.is_file()}
    if actual != set(expected) or len(expected) != bundle["file_count"]:
        raise ValueError(f"public cache has missing or extra files: {directory}")
    for relative, entry in expected.items():
        path = _inside(cache, relative)
        if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError(f"public cache payload changed: {directory}/{relative}")
    manifest_path = cache / "CACHE_MANIFEST.json"
    manifest_digest = _sha256(manifest_path)
    marker = (cache / "COMPLETE").read_text().strip()
    if marker not in (manifest_digest, f"{manifest_digest}  CACHE_MANIFEST.json"):
        raise ValueError("public cache completion identity differs from manifest")
    manifest = json.loads(manifest_path.read_text())
    return manifest, {
        "replay_contract": REPLAY_ID,
        "repository": release["repository"],
        "revision": release["revision"],
        "inventory_path": inventory_path,
        "inventory_sha256": inventory_digest,
        "archive_path": bundle["hosted_path"],
        "archive_sha256": bundle["sha256"],
        "manifest_path": f"{directory}/CACHE_MANIFEST.json",
        "manifest_sha256": manifest_digest,
        "verified_files": len(expected),
        "payload_transformation": bundle["payload_transformation"],
        "historical_identities_rebound": False,
    }


def development_freeze(root: Path, seed: int) -> tuple[dict, dict]:
    """Use the already-published normalized freeze, never forge an old digest."""
    root = Path(root)
    config_path = "configs/official_v6.yaml"
    config_digest = package_file_hash(root, config_path)
    official = yaml.safe_load((root / config_path).read_text())
    entries = [entry for entry in official["seeds"] if entry["seed"] == seed]
    if len(entries) != 1:
        raise ValueError(f"undeclared replay seed: {seed}")
    entry = entries[0]
    freeze_digest = package_file_hash(root, entry["freeze"])
    if freeze_digest != entry["freeze_sha256"]:
        raise ValueError("normalized development freeze does not match public protocol")
    freeze = json.loads((root / entry["freeze"]).read_text())
    for key, path_key, hash_key in (("checkpoint", "checkpoint", "checkpoint_sha256"),
                                    ("source_checkpoint", "source_checkpoint", "source_checkpoint_sha256")):
        if (freeze[key]["path"], freeze[key]["sha256"]) != (entry[path_key], entry[hash_key]):
            raise ValueError(f"development freeze {key} differs from public protocol")
    for arm in ("candidate", "source"):
        probe = freeze["matched_probe"]["probe_artifacts"][arm]
        if (probe["path"], probe["sha256"]) != (entry[f"{arm}_probe"], entry[f"{arm}_probe_sha256"]):
            raise ValueError("development freeze probe differs from public protocol")
    if freeze["matched_probe"]["frozen_floors"] != entry["frozen_train_probe_floors"]:
        raise ValueError("development freeze floors differ from public protocol")
    if freeze.get("protected_data_accessed") is not False or freeze.get("development_cache_opened") is not False:
        raise ValueError("development freeze was not recorded before protected access")
    return freeze, {
        "replay_contract": REPLAY_ID,
        "normalized_freeze_path": entry["freeze"],
        "normalized_freeze_sha256": freeze_digest,
        "public_protocol_path": config_path,
        "public_protocol_sha256": config_digest,
        "historical_identities_rebound": False,
        "new_training_or_probe_fitting": False,
    }
