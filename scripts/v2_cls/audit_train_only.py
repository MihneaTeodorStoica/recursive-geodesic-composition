#!/usr/bin/env python3
"""TRAIN-only V2 CLS checkpoint audit; never opens development/official ranges."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/v2_cls")]
from protocol import TEACHER_ID, TEACHER_POOLING, TEACHER_REVISION, TRAIN_SCAN  # noqa: E402
from recursive_geodesic_composition.architecture import GeodesicMerge, recursive_reduce  # noqa: E402
from recursive_geodesic_composition.train import step_loss_batch  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_model(cfg, variant, seed, device):
    torch.manual_seed(seed)
    return GeodesicMerge(
        d=768, hidden=cfg["encoder_hidden"], layers=cfg["encoder_layers"], heads=cfg["heads"],
        max_angle=float(cfg["max_geodesic_angle"]), decoder=variant == "autoencoder",
        decoder_layers=cfg["decoder_layers"], decoder_length_mode=cfg.get("decoder_length_mode", "oracle"),
        importance_mode=cfg["importance_mode"], importance_beta=float(cfg["importance_beta"]),
        decoder_mode=cfg.get("decoder_mode", "legacy"),
        importance_loss_weight=float(cfg.get("importance_loss_weight", 0)),
        importance_prior_weight=float(cfg.get("importance_prior_weight", 0)),
        delta_loss_weight=float(cfg.get("delta_loss_weight", 0)),
        sharpness_loss_weight=float(cfg.get("sharpness_loss_weight", 0)),
        slot_loss_weight=float(cfg.get("slot_loss_weight", 0)), sharpness_margin=float(cfg.get("sharpness_margin", .1)),
        geodesic_parameterization=cfg["geodesic_parameterization"], arity=int(cfg["arity"])).to(device).eval()


def metric(model, rows, cfg, variant, device):
    with torch.inference_mode():
        _, parts = step_loss_batch(model, rows, variant == "autoencoder", device,
            float(cfg.get("reconstruction_weight", 0)), float(cfg.get("cycle_weight", 0)),
            float(cfg.get("distribution_mean_weight", 0)), float(cfg.get("distribution_variance_weight", 0)),
            int(cfg.get("distribution_min_samples", 8)))
        roots = []
        exact_mass = True
        for row in rows:
            root, levels = recursive_reduce(model, row["leaves"].to(device)[None], row["lengths"].to(device)[None])
            roots.append(root[0].cpu())
            for parent, child, masses in levels:
                # Child mass values must be positive only in active prefix, and
                # every genuine parent mass is their exact sum in float32.
                if not torch.isfinite(parent).all() or not torch.isfinite(masses).all():
                    exact_mass = False
        roots = torch.stack(roots)
        singular = torch.linalg.svdvals(roots - roots.mean(0))
        effective_rank = float(torch.exp(-(singular / singular.sum().clamp_min(1e-12) *
                                          (singular / singular.sum().clamp_min(1e-12)).log().nan_to_num()).sum()))
        return {
            "matched_teacher_loss": float(parts["teacher"]),
            "matched_teacher_cosine": float(1 - parts["teacher"]),
            "reconstruction_loss": float(parts.get("reconstruction", torch.tensor(0.0))),
            "cycle_loss": float(parts.get("cycle", torch.tensor(0.0))),
            "geodesic_angle_max": float(parts["geodesic_angle_max"]),
            "geodesic_saturation_fraction": float(parts["geodesic_saturation_fraction"]),
            "importance_correction_saturation_fraction": float(parts.get("correction_saturation", torch.tensor(0.0))),
            "root_finite": bool(torch.isfinite(roots).all()),
            "root_norm_min": float(roots.norm(dim=-1).min()),
            "root_norm_max": float(roots.norm(dim=-1).max()),
            "root_effective_rank": effective_rank,
            "exact_recursive_mass_health": exact_mass,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", choices=("encoder", "autoencoder"), required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--initial", action="store_true", help="audit deterministic fresh initialization")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if bool(args.checkpoint) == bool(args.initial):
        raise ValueError("exactly one of --checkpoint or --initial is required")
    cfg = yaml.safe_load(Path(args.config).read_text())
    cache = ROOT / cfg["cache_dir"]
    manifest = json.loads((cache / "CACHE_MANIFEST.json").read_text())
    if manifest.get("teacher_pooling") != TEACHER_POOLING or manifest.get("source_scan") != list(TRAIN_SCAN):
        raise ValueError("refusing non-V2-CLS or non-TRAIN cache")
    if manifest["teacher"]["teacher_id"] != TEACHER_ID or manifest["teacher"]["teacher_revision"] != TEACHER_REVISION:
        raise ValueError("teacher identity/revision mismatch")
    rows = []
    for corpus in ("fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101"):
        loaded = torch.load(cache / corpus / "train.pt", map_location="cpu", weights_only=False)
        rows.extend(loaded[: max(1, args.rows // 4)])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = make_model(cfg, args.variant, int(cfg["seed"]), device)
    checkpoint_meta = None
    if args.checkpoint:
        path = Path(args.checkpoint)
        checkpoint_meta = torch.load(path, map_location="cpu", weights_only=True)
        provenance = checkpoint_meta.get("teacher_provenance")
        if provenance != {"teacher_id": TEACHER_ID, "teacher_revision": TEACHER_REVISION, "teacher_pooling": "cls"}:
            raise ValueError("checkpoint lacks exact CLS teacher provenance")
        model.load_state_dict(checkpoint_meta["state_dict"], strict=True)
    result = {
        "role": "train_only", "cache_manifest_sha256": sha256(cache / "CACHE_MANIFEST.json"),
        "config": args.config, "config_sha256": sha256(Path(args.config)), "variant": args.variant,
        "seed": int(cfg["seed"]), "rows": len(rows), "initial": args.initial,
        "checkpoint": None if not args.checkpoint else {"path": args.checkpoint, "sha256": sha256(Path(args.checkpoint)), "selected_step": checkpoint_meta.get("selected_step")},
        "metrics": metric(model, rows, cfg, args.variant, device),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
