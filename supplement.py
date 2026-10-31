#!/usr/bin/env python3
"""Offline supplementary integrity, saved-score replay, and exact-weight placement.

Uses only the Python standard library; never downloads data or executes training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes supplement: {relative}")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isfinite(actual) and abs(actual - expected) < 2e-13,
            f"{message}: {actual} != {expected}")


def verify():
    manifest = load("reproducibility/PACKAGE_MANIFEST.json")
    for entry in manifest["files"]:
        path = checked_path(entry["path"])
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"size mismatch: {entry['path']}")
        require(digest(path) == entry["sha256"], f"SHA-256 mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} packaged files by size and SHA-256.")
    print("Package manifest and SHA256SUMS are checksum containers, not self-authenticating signatures.")


def auc(labels, scores):
    require(len(labels) == len(scores) and len(labels) > 0, "invalid classifier arrays")
    require(set(labels) == {0, 1}, "classifier labels must contain both classes")
    require(all(math.isfinite(x) for x in scores), "nonfinite classifier scores")
    ordered = sorted(zip(scores, labels))
    negatives = sum(y == 0 for y in labels)
    positives = len(labels) - negatives
    numerator = 0.0
    below = 0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        positive_ties = sum(y for _, y in ordered[i:j])
        negative_ties = j - i - positive_ties
        numerator += positive_ties * (below + 0.5 * negative_ties)
        below += negative_ties
        i = j
    return numerator / (positives * negatives)


def replay_attacks():
    files = sorted((ROOT / "results/safe_baselines_v1/attacks").glob("*.json"))
    files += sorted((ROOT / "results/continuation_comparison_v1/attacks").glob("*.json"))
    require(bool(files), "no saved classifier predictions")
    bundles = 0
    for path in files:
        entry = json.loads(path.read_text())
        train = set(entry["classifier_train_groups"])
        held = set(entry["classifier_holdout_groups"])
        require(not train & held, f"classifier group leakage: {path.name}")
        folded = {}
        for family, bundle in entry["bundles"].items():
            require(len(bundle["groups"]) == len(bundle["y"]) == len(bundle["scores"]),
                    f"prediction array mismatch: {path.name}")
            pairs = defaultdict(list)
            for group, label in zip(bundle["groups"], bundle["y"]):
                pairs[group].append(label)
            require(set(pairs) == held and all(sorted(y) == [0, 1] for y in pairs.values()),
                    f"unpaired held-out predictions: {path.name}")
            raw = auc(bundle["y"], bundle["scores"])
            close(raw, entry["metrics"]["attacks"][family]["raw_auc"], f"raw AUC {path.name}/{family}")
            folded[family] = max(raw, 1 - raw)
            close(folded[family], entry["metrics"]["attacks"][family]["orientation_safe_auc"],
                  f"folded AUC {path.name}/{family}")
            bundles += 1
        close(max(folded.values()), entry["metrics"]["strongest_orientation_safe_auc"],
              f"strongest AUC {path.name}")
    print(f"Recomputed {bundles} classifier AUCs in {len(files)} saved prediction files; held-out pairs and TRAIN/holdout separation verified.")
    return len(files), bundles


def report():
    result = load("results/safe_baselines_v1/RESULTS.json")
    rows = load("results/safe_baselines_v1/DOCUMENT_SCORES.json")
    require(len(rows) == len({r['id'] for r in rows}) == 4000, "wrong safe-prefix population")
    require(Counter(r["active_children"] for r in rows) == {2: 2707, 3: 687, 4: 606},
            "wrong active-arity population")
    means = {}
    for method, expected in result["fidelity"]["three_seed_mean"]["pooled"]["methods"].items():
        values = [math.fsum(row["cosines"][f"{method}_seed{s}"] for s in (424242, 271828, 314159)) / 3
                  if method.startswith("learned_") else row["cosines"][method] for row in rows]
        means[method] = math.fsum(values) / len(values)
        close(means[method], expected["mean"], f"saved per-document fidelity {method}")
    for contrast, expected in result["fidelity"]["three_seed_mean"]["pooled"]["contrasts"].items():
        value = math.fsum(row["three_seed_mean_contrasts"][contrast] for row in rows) / len(rows)
        close(value, expected["mean"], f"saved paired contrast {contrast}")
    replay_attacks()
    print("\nSaved-result fidelity (recomputed from 4,000 per-document scores):")
    for method, value in means.items():
        print(f"  {method}: {value:.12f}")
    official = load("results/official_v6/results.json")["primary_three_seed_aggregate"]
    print(f"Historical official AUC (copied aggregate, not rerun): {official['mean_auc']:.12f}; CI95 {official['ci95']}")
    print("No classifier fitting, bootstrap interval replay, teacher encoding, model inference, timing benchmark or training was performed.")


def materialize(args):
    entries = load("reproducibility/WEIGHTS_MANIFEST.json")["artifacts"]
    roles = {entry["role"] for entry in entries}
    require(not set(args.role) - roles, f"unknown role; available: {sorted(roles)}")
    selected = [e for e in entries if not args.role or e["role"] in args.role]
    source = args.assets.resolve()
    # Verify all selected assets and existing destinations before making any write.
    placements = []
    for entry in selected:
        incoming = source / entry["asset_filename"]
        require(incoming.is_file(), f"missing external weight: {entry['asset_filename']}")
        require(incoming.stat().st_size == entry["byte_size"] and digest(incoming) == entry["sha256"],
                f"weight identity mismatch: {entry['asset_filename']}")
        targets = [entry["protocol_path"]]
        if entry["lineage"] == "historical-clean-v6":
            targets.append("artifacts/v1-paper/" + entry["asset_filename"])
        for relative in targets:
            destination = checked_path(relative)
            if destination.exists() or destination.is_symlink():
                require(destination.is_file() and digest(destination) == entry["sha256"],
                        f"refusing to replace different existing file: {relative}")
            placements.append((incoming, destination, entry))
    if not args.verify_only:
        for incoming, destination, entry in placements:
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + f".placing-{os.getpid()}")
            require(not temporary.exists(), f"stale placement file: {temporary.name}")
            try:
                if args.mode == "hardlink":
                    os.link(incoming, temporary)
                else:
                    shutil.copyfile(incoming, temporary)
                require(digest(temporary) == entry["sha256"], f"placement hash mismatch: {destination.name}")
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    print(f"Verified {len(selected)} exact weights; " +
          ("no files placed." if args.verify_only else f"placed/verified {len(placements)} protocol and lookup paths ({args.mode})."))
    print("Warm-checkpoint payloads contain personal paths: do not redistribute them as anonymous attachments.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify every listed packaged file")
    commands.add_parser("report", help="replay saved fidelity/AUC point estimates without dependencies")
    weights = commands.add_parser("materialize", help="verify exact local external weights, then place at required paths")
    weights.add_argument("assets", type=Path)
    weights.add_argument("--role", action="append", default=[])
    weights.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    weights.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "report":
        report()
    else:
        materialize(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
