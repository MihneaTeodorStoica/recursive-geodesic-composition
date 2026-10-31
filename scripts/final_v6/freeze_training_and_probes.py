"""Freeze all final checkpoints and TRAIN-only matched probes before development access."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from recursive_geodesic_composition.architecture import GeodesicMerge  # noqa: E402
from recursive_geodesic_composition.evaluate_kary_validation import ParentOnlyRetentionProbe, build_model  # noqa: E402
from protocol import (AGGREGATE_OUTPUT, CORPORA, DEV_CACHE, ROOT, SEEDS,  # noqa: E402
                      SOURCE_CHECKPOINTS, TRAIN_CACHE, atomic_json, load_config,
                      sha256_file, stable_hash)
from recursive_geodesic_composition.train import _is_decoder_parameter, _load_encoder_initialization  # noqa: E402


def parameter_change(reference, final, names):
    changed, squared, maximum, elements = 0, 0.0, 0.0, 0
    for name in names:
        if name not in reference or name not in final or reference[name].shape != final[name].shape:
            raise ValueError(f"parameter audit mismatch: {name}")
        delta = final[name].double() - reference[name].double()
        local = float(delta.abs().max()) if delta.numel() else 0.0
        changed += int(local > 0); squared += float((delta * delta).sum()); maximum = max(maximum, local); elements += delta.numel()
    return {"parameter_tensors": len(names), "changed_parameter_tensors": changed,
            "elements": elements, "l2_change": math.sqrt(squared),
            "maximum_absolute_change": maximum,
            "nonzero": bool(changed and squared > 0)}


def initial_model(cfg):
    random.seed(int(cfg["seed"])); torch.manual_seed(int(cfg["seed"]))
    model = GeodesicMerge(
        d=768, hidden=int(cfg["encoder_hidden"]), layers=int(cfg["encoder_layers"]),
        heads=int(cfg["heads"]), max_angle=float(cfg["max_geodesic_angle"]), decoder=True,
        decoder_layers=int(cfg["decoder_layers"]), decoder_length_mode=cfg["decoder_length_mode"],
        importance_mode=cfg["importance_mode"], importance_beta=float(cfg["importance_beta"]),
        decoder_mode=cfg["decoder_mode"], importance_loss_weight=float(cfg["importance_loss_weight"]),
        importance_prior_weight=float(cfg["importance_prior_weight"]),
        delta_loss_weight=float(cfg["delta_loss_weight"]), sharpness_loss_weight=float(cfg["sharpness_loss_weight"]),
        slot_loss_weight=float(cfg["slot_loss_weight"]), sharpness_margin=float(cfg["sharpness_margin"]),
        geodesic_parameterization=cfg["geodesic_parameterization"], arity=4)
    info = _load_encoder_initialization(
        model, ROOT / cfg["initialize_from_encoder_checkpoint"],
        cfg["initialize_from_encoder_checkpoint_sha256"], allow_autoencoder_source=True)
    if not info["continued_autoencoder_retraining"] or info["initialization_ignored_decoder_key_count"] < 1:
        raise ValueError("v6 must continue the embedded encoder while freshly initializing the decoder")
    return model, info


def select_probe_rows(rows_by, cfg):
    section = cfg["matched_probe"]
    fit, floor, audit = [], [], {}
    for corpus in CORPORA:
        by_group = {}
        for row in rows_by[corpus]:
            if int(row["leaves"].shape[0]) == 4 and row.get("segmentation") == "train":
                by_group.setdefault(str(row["id"]), row)
        ranked = sorted(by_group, key=lambda group: hashlib.sha256(
            f"v6-probe|{section['seed']}|{corpus}|{group}".encode()).digest())
        fit_count, floor_count = int(section["fit_rows_per_corpus"]), int(section["floor_rows_per_corpus"])
        if len(ranked) < fit_count + floor_count:
            raise ValueError(f"insufficient disjoint exact-K4 TRAIN groups for {corpus}")
        fit_groups, floor_groups = ranked[:fit_count], ranked[fit_count:fit_count + floor_count]
        fit.extend((corpus, by_group[group]) for group in fit_groups)
        floor.extend((corpus, by_group[group]) for group in floor_groups)
        audit[corpus] = {"fit_groups": len(fit_groups), "floor_groups": len(floor_groups),
                         "fit_sha256": stable_hash(fit_groups), "floor_sha256": stable_hash(floor_groups)}
    return fit, floor, audit


@torch.inference_mode()
def roots(model, rows, device, batch_size=128):
    values, dtype = [], next(model.parameters()).dtype
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        children = torch.stack([row["leaves"].float() for _, row in part]).to(device=device, dtype=dtype)
        masses = torch.stack([row["lengths"].float() for _, row in part]).to(device=device, dtype=dtype)
        values.append(model.merge(children, masses).float().cpu())
    return torch.cat(values)


def fit_probe(root_values, children, cfg, device, initial, schedule):
    section = cfg["matched_probe"]
    probe = ParentOnlyRetentionProbe(768, 4, int(section["hidden"])).to(device)
    probe.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=float(section["learning_rate"]), weight_decay=float(section["weight_decay"]))
    x, y = root_values.to(device), children.to(device)
    probe.train(); final = None
    for indexes in schedule:
        indexes = indexes.to(device)
        prediction = probe(x[indexes])
        cosine_loss = 1 - F.cosine_similarity(prediction, y[indexes], dim=-1).mean()
        mse = F.mse_loss(prediction, y[indexes]); loss = cosine_loss + float(section["mse_weight"]) * mse
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        final = {"loss": float(loss.detach()), "cosine_loss": float(cosine_loss.detach()), "mse": float(mse.detach())}
    probe.eval(); return probe, final


@torch.inference_mode()
def retention_scores(probe, root_values, children, labels, device):
    predictions = []
    for start in range(0, len(root_values), 512):
        predictions.append(probe(root_values[start:start + 512].to(device)).cpu())
    values = F.cosine_similarity(torch.cat(predictions), children, dim=-1).mean(-1).numpy()
    labels = np.asarray(labels)
    result = {"pooled": float(values.mean())}
    result.update({corpus: float(values[labels == corpus].mean()) for corpus in CORPORA})
    return result


def read_history(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def freeze_seed(seed, device, rows_by, fit_rows, floor_rows, selection_audit):
    cfg_path = ROOT / f"configs/final_v6/seed{seed}.yaml"; cfg = load_config(cfg_path)
    output = ROOT / cfg["output_dir"]
    checkpoint_path = output / f"autoencoder_seed{seed}.pt"
    history_path = output / f"autoencoder_seed{seed}.jsonl"
    freeze_path = output / "PRE_DEVELOPMENT_FROZEN.json"
    if freeze_path.exists(): raise FileExistsError(freeze_path)
    history = read_history(history_path)
    if len(history) != 400 or [row["step"] for row in history] != list(range(1, 401)):
        raise ValueError("training history is not exactly 400 consecutive updates")
    if any(bool(row["encoder_frozen"]) != (row["step"] <= 300) for row in history):
        raise ValueError("decoder/joint stage boundary changed")
    if any(any(int(row[f"corpus_batch_count/{corpus}"]) != 2 for corpus in CORPORA) for row in history):
        raise ValueError("equal deterministic corpus allocation changed")
    if any(not row["decoder_only_require_full_arity_rows"] for row in history):
        raise ValueError("decoder-only stage sampled a partial-arity row")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_path, source_sha = SOURCE_CHECKPOINTS[seed]
    if sha256_file(ROOT / source_path) != source_sha: raise ValueError("continued AE source hash mismatch")
    if checkpoint.get("variant") != "autoencoder" or checkpoint.get("selected_step") != 400:
        raise ValueError("checkpoint is not the fixed final strict AE")
    initial, init_info = initial_model(cfg)
    initial_state, final_state = initial.state_dict(), checkpoint["state_dict"]
    source_state = torch.load(ROOT / source_path, map_location="cpu", weights_only=True)["state_dict"]
    parameter_names = [name for name, _ in initial.named_parameters()]
    encoder_names = sorted(name for name in parameter_names if not _is_decoder_parameter(name))
    decoder_names = sorted(name for name in parameter_names if _is_decoder_parameter(name))
    encoder_change = parameter_change(source_state, final_state, encoder_names)
    decoder_change = parameter_change(initial_state, final_state, decoder_names)
    for change in (encoder_change, decoder_change):
        if not change["nonzero"] or change["changed_parameter_tensors"] != change["parameter_tensors"]:
            raise AssertionError("every encoder and decoder parameter tensor must change")
    del initial, checkpoint, source_state

    candidate, _ = build_model(checkpoint_path, device); source, _ = build_model(ROOT / source_path, device)
    fit_candidate, fit_source = roots(candidate, fit_rows, device), roots(source, fit_rows, device)
    floor_candidate, floor_source = roots(candidate, floor_rows, device), roots(source, floor_rows, device)
    fit_children = torch.stack([row["leaves"].float() for _, row in fit_rows])
    floor_children = torch.stack([row["leaves"].float() for _, row in floor_rows])
    floor_labels = [corpus for corpus, _ in floor_rows]
    section = cfg["matched_probe"]
    torch.manual_seed(int(section["seed"])); template = ParentOnlyRetentionProbe(768, 4, int(section["hidden"]))
    initial_probe = {name: value.detach().clone() for name, value in template.state_dict().items()}
    generator = torch.Generator(device="cpu").manual_seed(int(section["seed"]) + 1)
    schedule = torch.randint(len(fit_rows), (int(section["updates"]), int(section["batch_size"])), generator=generator)
    probes, fit_final = {}, {}
    for arm, fit_roots in (("source", fit_source), ("candidate", fit_candidate)):
        probes[arm], fit_final[arm] = fit_probe(fit_roots, fit_children, cfg, device, initial_probe, schedule)
    floor_scores = {
        "source": retention_scores(probes["source"], floor_source, floor_children, floor_labels, device),
        "candidate": retention_scores(probes["candidate"], floor_candidate, floor_children, floor_labels, device),
    }
    floors = {scope: value - float(section["floor_margin"]) for scope, value in floor_scores["source"].items()}
    probe_dir = output / "train_fitted_matched_probes"; probe_dir.mkdir(exist_ok=False)
    probe_artifacts = {}
    for arm, probe in probes.items():
        path = probe_dir / f"{arm}_parent_only.pt"
        torch.save({"state_dict": probe.cpu().state_dict(), "arm": arm, "fit_role": "fresh_v6_train",
                    "seed": seed, "selection_audit": selection_audit,
                    "schedule_sha256": hashlib.sha256(schedule.numpy().tobytes()).hexdigest()}, path)
        probe_artifacts[arm] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    result = {
        "study": cfg["study"], "development_only": True, "development_cache_opened": False,
        "protected_data_accessed": False, "selection": "fixed_final_step_400_without_validation",
        "continued_autoencoder_retraining_not_from_scratch": True,
        "source_checkpoint": {"path": source_path, "sha256": source_sha},
        "checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)), "sha256": sha256_file(checkpoint_path), "step": 400},
        "training": {"updates": 400, "decoder_only_updates": 300, "joint_updates": 100,
                     "encoder_change": encoder_change, "decoder_change": decoder_change,
                     "maximum_geodesic_saturation_fraction": max(float(r.get("geodesic_saturation_fraction", 0)) for r in history),
                     "maximum_importance_saturation_fraction": max(float(r.get("correction_saturation", 0)) for r in history),
                     "all_numeric_finite": all(math.isfinite(float(v)) for r in history for v in r.values() if isinstance(v, (int, float)))},
        "initialization": init_info,
        "matched_probe": {"fit_role": "fresh_v6_train", "development_updates": 0,
                          "tensor_identical_initialization": True, "identical_minibatch_schedule": True,
                          "selection_audit": selection_audit, "fit_final": fit_final,
                          "train_floor_scores": floor_scores, "frozen_floors": floors,
                          "floor_rule": "source_probe_disjoint_train_floor_score_minus_fixed_margin",
                          "probe_artifacts": probe_artifacts,
                          "schedule_sha256": hashlib.sha256(schedule.numpy().tobytes()).hexdigest()},
        "config": str(cfg_path.relative_to(ROOT)), "config_sha256": sha256_file(cfg_path),
    }
    atomic_json(freeze_path, result)
    (output / "PRE_DEVELOPMENT_FROZEN").write_text(f"{sha256_file(freeze_path)}  PRE_DEVELOPMENT_FROZEN.json\n")
    return result


def run(device_name="cuda"):
    aggregate = ROOT / AGGREGATE_OUTPUT
    if aggregate.exists(): raise FileExistsError(aggregate)
    if (ROOT / DEV_CACHE).exists(): raise RuntimeError("development cache was opened before all freezes")
    if not (ROOT / TRAIN_CACHE / "COMPLETE").is_file(): raise RuntimeError("fresh TRAIN cache incomplete")
    aggregate.mkdir()
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    rows_by = {corpus: torch.load(ROOT / TRAIN_CACHE / corpus / "train.pt", map_location="cpu", weights_only=False) for corpus in CORPORA}
    cfg = load_config(ROOT / "configs/final_v6/seed424242.yaml")
    fit_rows, floor_rows, audit = select_probe_rows(rows_by, cfg)
    if {(c, str(r["id"])) for c, r in fit_rows} & {(c, str(r["id"])) for c, r in floor_rows}:
        raise AssertionError("probe fit/floor TRAIN groups overlap")
    for seed in SEEDS:
        freeze_seed(seed, device, rows_by, fit_rows, floor_rows, audit)
    declaration = {"study": AGGREGATE_OUTPUT.removeprefix("results_"), "development_only": True,
                   "development_cache_opened": False, "protected_data_accessed": False,
                   "all_three_fixed_before_development": True,
                   "seed_freezes": {str(seed): {"path": f"results_kary4_strict_autoencoder_v6_clean_development_seed{seed}/PRE_DEVELOPMENT_FROZEN.json",
                                                "sha256": sha256_file(ROOT / f"results_kary4_strict_autoencoder_v6_clean_development_seed{seed}/PRE_DEVELOPMENT_FROZEN.json")} for seed in SEEDS},
                   "promotion_gate": {"mean_strongest_auc_strictly_below": .60,
                                      "every_seed_strongest_auc_strictly_below": .61,
                                      "teacher_cosine_strictly_above": .98,
                                      "builtin_child_cosine_minimum": .90,
                                      "retention_floors_frozen": True,
                                      "all_encoder_decoder_parameters_changed": True,
                                      "no_collapse_or_saturation": True}}
    atomic_json(aggregate / "PRE_DEVELOPMENT_FROZEN.json", declaration)
    (aggregate / "PRE_DEVELOPMENT_FROZEN").write_text(f"{sha256_file(aggregate / 'PRE_DEVELOPMENT_FROZEN.json')}  PRE_DEVELOPMENT_FROZEN.json\n")
    return aggregate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); print(run(args.device))
