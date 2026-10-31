#!/usr/bin/env python3
"""Replay all three official-v6 seeds from the pinned public cache.

This is a new cached evaluation, not the original test collection. Both complete
4000-document arms and all fixed attacks/2000-bootstrap aggregates are mandatory.
Results are created only in replays/official_v6; existing outputs are refused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts", ROOT / "scripts/official_v6"):
    sys.path.insert(0, str(directory))

from evaluate_official_test_v6_recovery import evaluate
from official_test_v6_protocol import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = evaluate(args.config, args.device, replay=True)
    print(f"Official cached replay complete: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
