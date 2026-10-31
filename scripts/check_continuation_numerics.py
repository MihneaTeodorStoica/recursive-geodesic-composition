#!/usr/bin/env python3
"""Supplementary post-result numerical check; never replaces primary outputs.

Motivation: the observed teacher-cosine contrast is smaller than 1e-6. Replay
BOTH endpoints with B=17 and report, rather than suppress, numerical shifts.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from evaluate_continuation_comparison import CONFIG, load_protocol, load_inference, inputs
from safe_baselines_protocol import ROOT, atomic_json, configure_device, require_hash, sha256
from recursive_geodesic_composition.evaluate_kary_validation import build_model


def check(device_name):
    cfg, calibration, baseline, _, published = load_protocol(CONFIG)
    device = configure_device(cfg, device_name)
    root = ROOT / cfg["output_dir"]
    if (root / "NUMERICAL_CHECK_PROTOCOL.json").exists():
        raise FileExistsError("supplementary check is create-only")
    payload, manifest = load_inference(cfg)
    _, _, rows, _, _, _ = inputs(calibration, baseline, published)
    protocol = {"role": "supplementary_post_result_numerical_validation", "not_a_new_primary_analysis": True,
                "motivation": "observed teacher contrast below 1e-6", "implementation_sha256": sha256(__file__),
                "results_sha256": sha256(root / "RESULTS.json"), "inference_manifest_sha256": sha256(root / "INFERENCE_MANIFEST.json"),
                "batch_size": 17, "packing": "document_order_mixed_arity", "seeds": cfg["seeds"],
                "retain_all_shifts_without_result_substitution": True, "started_unix": time.time()}
    atomic_json(root / "NUMERICAL_CHECK_PROTOCOL.json", protocol)
    points = json.loads((root / "POINTS.json").read_text())
    outcomes, delta_shifts = {}, []
    with torch.inference_mode():
        for seed in map(str, cfg["seeds"]):
            stage_scores, outcomes[seed] = {}, {}
            for stage in ("source", "final"):
                entry = manifest["artifacts"][seed][stage]
                require_hash(ROOT / entry["path"], entry["sha256"])
                model, checkpoint = build_model(ROOT / entry["path"], device)
                del checkpoint
                parents = []
                for start in range(0, len(rows), 17):
                    part = rows[start:start+17]
                    x, n = torch.zeros(len(part), 4, 768, device=device), torch.zeros(len(part), 4, device=device)
                    for i, row in enumerate(part):
                        k = len(row["lengths"])
                        x[i, :k], n[i, :k] = row["leaves"].to(device), row["lengths"].to(device)
                    parents.append(model.merge(x, n).cpu())
                parents = torch.cat(parents)
                del model
                vector_error = float((parents - payload[f"{stage}_parents"][seed]).abs().max())
                assert vector_error <= cfg["independent_replay_max_absolute_tolerance"]
                stage_scores[stage] = F.cosine_similarity(parents.double(), payload["targets"].double(), dim=-1).numpy()
                shift = stage_scores[stage] - points["teacher"][seed][stage]
                outcomes[seed][stage] = {"maximum_absolute_vector_error": vector_error, "maximum_absolute_document_score_shift": float(abs(shift).max()), "mean_score_shift": float(shift.mean())}
                require_hash(ROOT / entry["path"], entry["sha256"])
            shift = stage_scores["final"] - stage_scores["source"] - points["teacher"][seed]["delta"]
            outcomes[seed]["delta"] = {"maximum_absolute_document_shift": float(abs(shift).max()), "mean_shift": float(shift.mean())}
            delta_shifts.append(shift)
    mean_shift = np.mean(delta_shifts, axis=0)
    require_hash(root / "RESULTS.json", protocol["results_sha256"])
    result = protocol | {"outcomes": outcomes, "three_seed_mean_delta_shift": float(mean_shift.mean()),
                         "maximum_absolute_checkpoint_mean_document_delta_shift": float(abs(mean_shift).max()),
                         "completed_unix": time.time()}
    atomic_json(root / "NUMERICAL_CHECK.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    check(parser.parse_args().device)
