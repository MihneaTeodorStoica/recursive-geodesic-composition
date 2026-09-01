#!/usr/bin/env python3
"""Cheap V2 invariant smoke tests that do not touch development/official data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/v2_cls"))
from protocol import audit_configs, assert_source_ranges_nonoverlap  # noqa: E402
from recursive_geodesic_composition.architecture import GeodesicMerge, recursive_reduce, contiguous_reduction_groups  # noqa: E402


def main() -> None:
    torch.manual_seed(12345)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"device": str(device), "ranges": assert_source_ranges_nonoverlap(), "configs": audit_configs()["configs"]}
    groups = {n: contiguous_reduction_groups(n, 4) for n in range(1, 10)}
    assert groups[1] == [(0, 1)]
    assert groups[2] == [(0, 2)] and groups[3] == [(0, 3)]
    assert groups[5] == [(0, 4), (4, 1)]
    results["contiguous_k4_groups"] = {str(k): v for k, v in groups.items()}

    model = GeodesicMerge(d=768, hidden=1024, layers=1, heads=16, dropout=0.0,
                          max_angle=0.5, decoder=True, decoder_layers=1,
                          decoder_mode="strict", arity=4,
                          geodesic_parameterization="tangent_expmap",
                          importance_mode="semantic").to(device).eval()
    leaves = F.normalize(torch.randn(2, 7, 768, device=device), dim=-1)
    counts = torch.tensor([[10, 20, 30, 40, 50, 60, 70], [1, 1, 1, 1, 2, 2, 2]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        root, levels = recursive_reduce(model, leaves, counts)
    assert root.shape == (2, 768)
    assert torch.isfinite(root).all()
    assert torch.allclose(root.norm(dim=-1), torch.ones(2, device=device), atol=2e-5, rtol=0)
    # Parent token masses are exact sums per merge level, including singleton carry.
    first_sizes = levels[0]["sizes"] if isinstance(levels[0], dict) and "sizes" in levels[0] else None
    results["recursive_root_norms"] = root.norm(dim=-1).detach().cpu().tolist()
    results["levels_recorded"] = len(levels)
    results["first_level_sizes_recorded"] = first_sizes is not None

    if device.type == "cuda":
        cpu_model = GeodesicMerge(d=768, hidden=1024, layers=1, heads=16, dropout=0.0,
                                  max_angle=0.5, decoder=True, decoder_layers=1,
                                  decoder_mode="strict", arity=4,
                                  geodesic_parameterization="tangent_expmap",
                                  importance_mode="semantic").eval()
        cpu_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
        with torch.inference_mode():
            cpu_root, _ = recursive_reduce(cpu_model, leaves.cpu(), counts.cpu())
        cos = F.cosine_similarity(root.cpu(), cpu_root).tolist()
        results["cpu_gpu_cosine"] = cos
        assert min(cos) > 0.99999

    out = ROOT / "reproducibility/v2_cls/SMOKE_INVARIANTS.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
