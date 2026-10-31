#!/usr/bin/env python3
"""Verify downloaded v1-paper assets and place them at frozen protocol paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reproducibility/ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_destination(relative: str) -> Path:
    destination = (ROOT / relative).resolve(strict=False)
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"protocol path escapes repository: {relative}") from exc
    return destination


def verify_asset(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing release asset: {path}")
    actual_size = path.stat().st_size
    if actual_size != int(entry["byte_size"]):
        raise ValueError(
            f"size mismatch for {path.name}: {actual_size} != {entry['byte_size']}"
        )
    actual_hash = sha256(path)
    if actual_hash != entry["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: {actual_hash} != {entry['sha256']}"
        )


def materialize(source: Path, destination: Path, mode: str) -> str:
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and sha256(destination) == sha256(source):
            return "already verified"
        raise FileExistsError(f"refusing to replace existing path: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.materializing-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary path already exists: {temporary}")
    try:
        if mode == "copy":
            shutil.copyfile(source, temporary)
        elif mode == "hardlink":
            os.link(source, temporary)
        elif mode == "symlink":
            temporary.symlink_to(source.resolve())
        else:  # guarded by argparse
            raise ValueError(f"unknown mode: {mode}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset_dir",
        type=Path,
        help="directory populated by `gh release download v1-paper`",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="materialize only this manifest role (repeatable; default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="placement method (default: copy)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify selected assets without creating protocol paths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_dir = args.asset_dir.resolve()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["artifacts"]
    available_roles = {entry["role"] for entry in entries}
    requested_roles = set(args.role)
    unknown = requested_roles - available_roles
    if unknown:
        raise ValueError(
            f"unknown roles {sorted(unknown)}; available roles: {sorted(available_roles)}"
        )
    selected = [
        entry
        for entry in entries
        if not requested_roles or entry["role"] in requested_roles
    ]
    if not selected:
        raise ValueError("no release artifacts selected")

    downloaded_manifest = asset_dir / MANIFEST_PATH.name
    if downloaded_manifest.exists() and downloaded_manifest.read_bytes() != MANIFEST_PATH.read_bytes():
        raise ValueError("downloaded release manifest differs from the repository copy")

    verified: list[tuple[Path, Path, dict]] = []
    for entry in selected:
        source = asset_dir / entry["asset_filename"]
        verify_asset(source, entry)
        destination = checked_destination(entry["protocol_path"])
        verified.append((source, destination, entry))
        print(
            f"verified {source.name}: role={entry['role']} seed={entry['seed']} "
            f"size={entry['byte_size']} sha256={entry['sha256']}"
        )

    if args.verify_only:
        print(f"Verified {len(verified)} release artifacts; no files materialized.")
        return 0

    for source, destination, _entry in verified:
        action = materialize(source, destination, args.mode)
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"materialized hash mismatch: {destination}")
        print(f"{action}: {destination.relative_to(ROOT)}")
    print(f"Materialized {len(verified)} verified release artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
