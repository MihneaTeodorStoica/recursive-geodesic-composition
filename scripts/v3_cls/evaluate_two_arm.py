#!/usr/bin/env python3
"""Run the cache-only V3 CLS two-arm evaluator (never prepares data or trains)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from recursive_geodesic_composition.evaluate_v3_cls import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Hash-pinned role/checkpoint/cache evaluation specification")
    parser.add_argument("--protocol", default=str(ROOT / "configs/v3_cls/evaluator_two_arm.yaml"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(run(ROOT, Path(args.protocol).resolve(), Path(args.spec).resolve(), args.device))


if __name__ == "__main__":
    main()
