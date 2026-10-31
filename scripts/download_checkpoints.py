#!/usr/bin/env python3
"""Download selected, pinned public checkpoints into the frozen protocol paths.

Standard library only. No authentication headers or cached credentials are used.
Every reused/downloaded file is checked by size and SHA-256 before placement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes destination: {relative}")
    return path


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["byte_size"]:
        raise ValueError(f"missing file or size mismatch: {path.name}")
    if digest(path) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def place(source, destination, entry):
    if destination.exists():
        verify(destination, entry)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            verify(temporary, entry)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True,
                        help="manifest group, repeatable; use 'all' explicitly for every checkpoint")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="writable evaluation checkout root (default: this checkout)")
    parser.add_argument("--verify", action="store_true",
                        help="also recheck every installed protocol/lookup path after placement")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reproducibility/HOSTED_CHECKPOINTS.json").read_text())
    revision = manifest["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("manifest must pin an immutable 40-hex Hugging Face revision")
    groups = manifest["groups"]
    unknown = set(args.group) - set(groups) - {"all"}
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}; available: {sorted(groups)} and all")
    requested = {name for group in args.group if group != "all" for name in groups[group]}
    selected = [entry for entry in manifest["artifacts"]
                if "all" in args.group or entry["asset_filename"] in requested]
    if not selected:
        raise ValueError("selected group has no publishable checkpoints; inspect manifest exclusions")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    installed = []
    downloaded = 0
    for entry in selected:
        targets = [checked_path(root, path) for path in entry["install_paths"]]
        existing = [path for path in targets if path.exists()]
        for path in existing:
            verify(path, entry)
        source = existing[0] if existing else None
        bundled = ROOT / "weights" / entry["asset_filename"]
        if source is None and bundled.is_file():
            verify(bundled, entry)
            source = bundled
        if source is None:
            destination = targets[0]
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
            os.close(fd)
            temporary = Path(name)
            url = (f"https://huggingface.co/{manifest['repo_id']}/resolve/{revision}/"
                   + quote(entry["hosted_path"], safe="/"))
            try:
                # Deliberately do not use an authenticated Hub client or environment token.
                request = Request(url, headers={"User-Agent": "anonymous-checkpoint-downloader/1"})
                with urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=8 << 20)
                verify(temporary, entry)
                os.replace(temporary, destination)
                source = destination
                downloaded += 1
            finally:
                temporary.unlink(missing_ok=True)
        for destination in targets:
            if destination != source:
                place(source, destination, entry)
            installed.append((destination, entry))
        print(f"verified {entry['asset_filename']}: {entry['sha256']}")
    if args.verify:
        for path, entry in installed:
            verify(path, entry)
    print(f"Verified {len(selected)} checkpoints; downloaded {downloaded}; "
          f"installed/verified {len(installed)} protocol/lookup paths at revision {revision}.")
    print("Historical experiment hashes are unchanged. Metadata-sanitized warm files have distinct "
          "hashes and do not satisfy original-byte provenance guards; see CHECKPOINT_ACCESS.md.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
