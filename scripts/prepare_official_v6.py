#!/usr/bin/env python3
"""Run the frozen official-v6 two-arm cache preparer."""

import runpy
import sys
from pathlib import Path

stage = Path(__file__).resolve().parent / "official_v6"
sys.path.insert(0, str(stage))
runpy.run_path(str(stage / "prepare_official_test_v6.py"), run_name="__main__")
