#!/usr/bin/env python3
"""Replay every published development seed from public caches, without training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from final_v6.evaluate_development import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", action="append", type=int, choices=(424242, 271828, 314159),
                        help="Select a seed; default runs all three, in published order")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    seeds = args.seed or (424242, 271828, 314159)
    if len(seeds) != len(set(seeds)):
        parser.error("do not request the same create-only seed twice")
    for seed in seeds:
        path = ROOT / f"configs/final_v6/seed{seed}.yaml"
        print(run(path, args.device, replay=True), flush=True)


if __name__ == "__main__":
    main()
