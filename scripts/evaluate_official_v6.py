#!/usr/bin/env python3
"""Run the frozen official-v6 evaluator-only recovery."""

import runpy
import sys
from pathlib import Path

stage = Path(__file__).resolve().parent / "official_v6"
sys.path.insert(0, str(stage))
runpy.run_path(str(stage / "evaluate_official_test_v6_recovery.py"), run_name="__main__")
