#!/usr/bin/env python3
"""Verify and install ONLY new v2 review weights/PDF, never v1-paper models."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"reproducibility/REVIEW_RELEASE_MANIFEST.json"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets",type=Path,help="directory created by 'gh release download v2-paper-review'")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only",action="store_true")
    group.add_argument("--mode",choices=("hardlink","copy"))
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text())
    assert manifest["release"]["tag"]=="v2-paper-review" and manifest["release"]["previous_historical_release"]=="v1-paper"
    recorded=manifest["checksum_asset"];checksum=args.assets/recorded["filename"]
    if not checksum.is_file() or checksum.stat().st_size!=recorded["byte_size"] or digest(checksum)!=recorded["sha256"]:
        raise ValueError("missing or corrupted release checksum asset")
    expected="".join(f"{entry['sha256']}  {entry['asset_filename']}\n" for entry in manifest["published_assets"])
    if checksum.read_text()!=expected:raise ValueError("release checksum list differs from pinned manifest")
    for entry in manifest["published_assets"]:
        incoming=args.assets/entry["asset_filename"]
        if not incoming.is_file() or incoming.stat().st_size!=entry["byte_size"] or digest(incoming)!=entry["sha256"]:
            raise ValueError(f"missing or corrupted release asset: {incoming}")
        if not args.verify_only:
            output=ROOT/entry["protocol_path"]
            if output.is_file():
                if output.stat().st_size!=entry["byte_size"] or digest(output)!=entry["sha256"]:
                    raise FileExistsError(f"existing target is not the pinned artifact: {output}")
            else:
                output.parent.mkdir(parents=True,exist_ok=True)
                temporary=output.with_name(output.name+".staging.review")
                if temporary.exists():raise FileExistsError(temporary)
                if args.mode=="hardlink":os.link(incoming,temporary)
                else:shutil.copyfile(incoming,temporary)
                os.replace(temporary,output)
                assert digest(output)==entry["sha256"]
    print(f"Verified {len(manifest['published_assets'])} new review weights/PDF; mode={args.mode or 'verify-only'}. No v1-paper assets changed.")


if __name__=="__main__":main()
