from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_replay import public_cache_manifest


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _released_cache(root):
    cache = root / "cache_public"
    cache.mkdir()
    (cache / "leaves.bin").write_bytes(bytes(range(64)))
    manifest = cache / "CACHE_MANIFEST.json"
    manifest.write_text(json.dumps({"complete": True, "scientific_role": "test"}))
    (cache / "COMPLETE").write_text(f"{_digest(manifest)}  CACHE_MANIFEST.json\n")
    files = [{"path": "cache_public/" + p.name, "bytes": p.stat().st_size,
              "sha256": _digest(p)} for p in cache.iterdir()]
    bundle = {"cache_directory": "cache_public", "file_count": len(files), "files": files,
              "hosted_path": "caches/cache_public.tar.gz", "sha256": "archive-identity",
              "payload_transformation": "none"}
    release = {"repository": "anonymous/artifacts", "revision": "public-release",
               "inventory": {"archives": [bundle]}}
    (root / "reproducibility").mkdir()
    inventory = root / "reproducibility/HOSTED_CACHES.json"
    inventory.write_text(json.dumps(release))
    (root / "reproducibility/PACKAGE_MANIFEST.json").write_text(json.dumps({"files": [
        {"path": "reproducibility/HOSTED_CACHES.json", "bytes": inventory.stat().st_size,
         "sha256": _digest(inventory)}]}))
    return cache


def test_replay_rejects_modified_payload_even_with_intact_local_completion(tmp_path):
    cache = _released_cache(tmp_path)
    (cache / "leaves.bin").write_bytes(b"different scientific payload")
    with pytest.raises(ValueError, match="public cache payload changed"):
        public_cache_manifest(tmp_path, "cache_public")


def test_replay_rejects_self_consistent_but_unreleased_metadata(tmp_path):
    cache = _released_cache(tmp_path)
    manifest = cache / "CACHE_MANIFEST.json"
    manifest.write_text(json.dumps({"complete": True, "scientific_role": "train"}))
    (cache / "COMPLETE").write_text(f"{_digest(manifest)}  CACHE_MANIFEST.json\n")
    with pytest.raises(ValueError, match="public cache payload changed"):
        public_cache_manifest(tmp_path, "cache_public")


@pytest.mark.parametrize("change", ["missing", "extra", "symlink"])
def test_replay_rejects_changed_cache_membership(tmp_path, change):
    cache = _released_cache(tmp_path)
    if change == "missing":
        (cache / "leaves.bin").unlink()
    elif change == "extra":
        (cache / "undeclared.bin").write_bytes(b"extra")
    else:
        payload = cache / "leaves.bin"
        saved = tmp_path / "outside.bin"
        payload.rename(saved)
        payload.symlink_to(saved)
    with pytest.raises(ValueError, match="missing or extra|symbolic link"):
        public_cache_manifest(tmp_path, "cache_public")
