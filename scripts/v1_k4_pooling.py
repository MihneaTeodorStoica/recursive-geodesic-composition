#!/usr/bin/env python3
"""Additive matched parent-only V1 K4-versus-pooling retention experiment.

No command trains or modifies V1. Probe fitting reads only the frozen clean-v6
TRAIN cache. Evaluation reads only the already-spent official-v6 safe population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recursive_geodesic_composition.evaluate_kary_validation import (  # noqa: E402
    ParentOnlyRetentionProbe,
    build_model,
)

CONFIG = ROOT / "configs/v1_k4_pooling.yaml"
METHODS = ("v1", "weighted", "mean")
PUBLISHED_OFFICIAL_RESULTS_SHA256 = "2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("study") != "v1_k4_matched_pooling" or cfg.get("additive_experiment") is not True:
        raise ValueError("not the frozen additive V1 pooling protocol")
    if int(cfg.get("primary_v1_seed", -1)) != 424242:
        raise ValueError("primary frozen V1 seed changed")
    ranges = cfg["source_ranges"]
    expected = {
        "train": [140000, 152000], "development": [152000, 161000],
        "buffer": [161000, 162000], "official": [162000, 192000],
    }
    if ranges != expected:
        raise ValueError("clean-v6 source-role ranges changed")
    probe = cfg["probe"]
    fixed = {
        "architecture": "ParentOnlyRetentionProbe", "dimension": 768, "arity": 4,
        "hidden": 512, "fit_rows_per_corpus": 80, "initialization_seed": 730241,
        "schedule_seed": 730242, "updates": 400, "batch_size": 128,
        "optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.001,
        "gradient_clip_norm": 1.0, "loss": "cosine_plus_weighted_mse", "mse_weight": 0.25,
    }
    if probe != fixed or cfg["bootstrap"] != {"replicates": 2000, "seed": 118731}:
        raise ValueError("matched probe/bootstrap protocol changed")
    expected_paths = {
        "checkpoint": "results_kary4_strict_autoencoder_v6_clean_development_seed424242/autoencoder_seed424242.pt",
        "train_cache": "cache_kary4_strict_autoencoder_v6_clean_train",
        "official_cache_preferred": "cache_kary4_strict_autoencoder_official_test_v6",
        "official_cache_additive": "cache_v1_k4_matched_pooling_official_safe",
        "output_dir": "results/v1_k4_matched_pooling",
    }
    if any(cfg.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("frozen artifact namespace changed")
    pins = (
        ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"),
        ("fineweb", "HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc"),
        ("openwebtext", "Skylion007/openwebtext", None, "79d93d786212f7344586290adb811d4ae6a1762c"),
        ("wikipedia_en_20231101", "wikimedia/wikipedia", "20231101.en", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"),
    )
    actual_pins = tuple((item["id"], item["dataset_name"], item.get("dataset_config"), item["dataset_revision"]) for item in cfg["corpora"])
    exact_hashes = {
        "fineweb_edu": "d21ba0352b719de1014ed2baf90fe363078899e868f849cd368c6f46d6c2e751",
        "fineweb": "806856dfaf4fad26e38217db9206310d291e5f6dae0963cadab05dbf9e995b01",
        "openwebtext": "a146c237e699f91ae389baa23013be674f4e480a8a9f014f66f2977b5daa5b4d",
        "wikipedia_en_20231101": "f0bf70d773027906896cb0c70dc7b03624fca195a961ae183956026054ab3238",
    }
    if actual_pins != pins or int(cfg["official_population"].get("expected_exact_k4_total", -1)) != 606 or cfg["official_population"].get("expected_exact_k4_order_sha256") != exact_hashes:
        raise ValueError("corpus pins or official population changed")
    if int(cfg.get("official_updates", -1)) != 0:
        raise ValueError("official fitting is forbidden")
    return cfg


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing silent backend change")
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    return device


def _normal_children(row: dict) -> torch.Tensor:
    children = row["leaves"].float()
    if tuple(children.shape) != (4, 768) or not torch.isfinite(children).all():
        raise ValueError("expected finite exact-K4 child embeddings")
    return F.normalize(children, dim=-1)


def pooling_parents(children: torch.Tensor, masses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token-mass weighted and unweighted normalized means."""
    if children.ndim != 3 or masses.shape != children.shape[:2]:
        raise ValueError("pool inputs must be [batch,K,D] and [batch,K]")
    children = F.normalize(children.float(), dim=-1)
    masses = masses.float()
    if not torch.isfinite(children).all() or not torch.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("pool inputs must be finite with positive masses")
    q = masses / masses.sum(-1, keepdim=True)
    weighted = F.normalize((q[..., None] * children).sum(1), dim=-1)
    mean = F.normalize(children.sum(1), dim=-1)
    return weighted, mean


def select_train_rows(rows_by: dict[str, list], cfg: dict) -> tuple[list[tuple[str, dict]], dict]:
    """Reproduce the V1 TRAIN-probe fit-group selection exactly."""
    section, selected, audit = cfg["probe"], [], {}
    for corpus_cfg in cfg["corpora"]:
        corpus = corpus_cfg["id"]
        by_group = {}
        for row in rows_by[corpus]:
            if int(row["leaves"].shape[0]) == 4 and row.get("segmentation") == "train":
                by_group.setdefault(str(row["id"]), row)
        ranked = sorted(
            by_group,
            key=lambda group: hashlib.sha256(
                f"v6-probe|{section['initialization_seed']}|{corpus}|{group}".encode()
            ).digest(),
        )
        groups = ranked[: int(section["fit_rows_per_corpus"])]
        if len(groups) != int(section["fit_rows_per_corpus"]):
            raise ValueError(f"insufficient exact-K4 TRAIN groups: {corpus}")
        selected.extend((corpus, by_group[group]) for group in groups)
        audit[corpus] = {"rows": len(groups), "group_order_sha256": stable_hash(groups)}
    return selected, audit


def _load_train(cfg: dict) -> tuple[dict[str, list], dict]:
    cache = ROOT / cfg["train_cache"]
    manifest = json.loads((cache / "CACHE_MANIFEST.json").read_text())
    manifest_hash = sha256_file(cache / "CACHE_MANIFEST.json")
    if (cache / "COMPLETE").read_text().strip() != f"{manifest_hash}  CACHE_MANIFEST.json":
        raise ValueError("TRAIN cache COMPLETE marker does not bind its manifest")
    if manifest.get("role") != "train" or manifest.get("source_scan") != cfg["source_ranges"]["train"]:
        raise ValueError("cache is not the frozen clean-v6 TRAIN role/range")
    if manifest.get("protected_data_accessed") is not False or manifest.get("development_only") is not True or manifest.get("next_development_opened") is not False:
        raise ValueError("TRAIN cache provenance permits protected access")
    if manifest.get("config_sha256") != sha256_file(ROOT / "configs/final_v6/seed424242.yaml"):
        raise ValueError("TRAIN cache construction config changed")
    expected_corpora = [item["id"] for item in cfg["corpora"]]
    if manifest.get("corpora") != expected_corpora:
        raise ValueError("TRAIN cache corpus order changed")
    rows = {}
    for corpus in expected_corpora:
        relative = f"{corpus}/train.pt"
        path = cache / relative
        if sha256_file(path) != manifest.get("artifacts", {}).get(relative, {}).get("sha256"):
            raise ValueError(f"TRAIN shard hash mismatch: {corpus}")
        rows[corpus] = torch.load(path, map_location="cpu", weights_only=False)
    return rows, {"path": cfg["train_cache"], "manifest_sha256": sha256_file(cache / "CACHE_MANIFEST.json")}


@torch.inference_mode()
def parent_features(model, selected: list[tuple[str, dict]], device: torch.device, batch_size: int = 128):
    values = {method: [] for method in METHODS}
    targets = []
    dtype = next(model.parameters()).dtype
    for start in range(0, len(selected), batch_size):
        part = selected[start : start + batch_size]
        children = torch.stack([_normal_children(row) for _, row in part])
        masses = torch.stack([row["lengths"].float() for _, row in part])
        weighted, mean = pooling_parents(children, masses)
        v1 = model.merge(children.to(device=device, dtype=dtype), masses.to(device=device, dtype=dtype))
        values["v1"].append(v1.float().cpu())
        values["weighted"].append(weighted)
        values["mean"].append(mean)
        targets.append(children)
    return {key: torch.cat(value) for key, value in values.items()}, torch.cat(targets)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit_one(features, targets, cfg, initial, schedule, device):
    section = cfg["probe"]
    probe = ParentOnlyRetentionProbe(768, 4, 512).to(device)
    probe.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(section["learning_rate"]), weight_decay=float(section["weight_decay"])
    )
    x, y = features.to(device), targets.to(device)
    probe.train()
    final = None
    for indexes in schedule:
        indexes = indexes.to(device)
        prediction = probe(x[indexes])
        cosine_loss = 1 - F.cosine_similarity(prediction, y[indexes], dim=-1).mean()
        mse = F.mse_loss(prediction, y[indexes])
        loss = cosine_loss + float(section["mse_weight"]) * mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            probe.parameters(), float(section["gradient_clip_norm"]), error_if_nonfinite=True
        )
        optimizer.step()
        final = {"loss": float(loss.detach()), "cosine_loss": float(cosine_loss.detach()), "mse": float(mse.detach())}
    return probe.cpu().eval(), final


def fit(config_path: Path, device_name: str) -> Path:
    cfg = load_config(config_path)
    output = ROOT / cfg["output_dir"]
    probe_dir = output / "probes"
    if probe_dir.exists():
        raise FileExistsError("additive probe namespace is create-only")
    checkpoint = ROOT / cfg["checkpoint"]
    before = sha256_file(checkpoint)
    if before != cfg["checkpoint_sha256"]:
        raise ValueError("frozen V1 checkpoint hash mismatch")
    rows_by, train_provenance = _load_train(cfg)
    selected, selection = select_train_rows(rows_by, cfg)
    train_ids = [f"{corpus}:{row['id']}" for corpus, row in selected]
    device = resolve_device(device_name)
    model, _ = build_model(checkpoint, device)
    features, targets = parent_features(model, selected, device)
    del model
    torch.manual_seed(int(cfg["probe"]["initialization_seed"]))
    template = ParentOnlyRetentionProbe(768, 4, 512)
    initial = {name: value.detach().clone() for name, value in template.state_dict().items()}
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["probe"]["schedule_seed"]))
    schedule = torch.randint(
        len(selected),
        (int(cfg["probe"]["updates"]), int(cfg["probe"]["batch_size"])),
        generator=generator,
    )
    probe_dir.mkdir(parents=True)
    artifacts, finals = {}, {}
    initialization_hash = tensor_state_sha256(initial)
    schedule_hash = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    for method in METHODS:
        probe, finals[method] = fit_one(features[method], targets, cfg, initial, schedule, device)
        path = probe_dir / f"{method}_parent_only.pt"
        torch.save(
            {
                "state_dict": probe.state_dict(), "representation": method,
                "fit_role": "clean_v6_train", "source_range": cfg["source_ranges"]["train"],
                "v1_seed": cfg["primary_v1_seed"], "initialization_sha256": initialization_hash,
                "schedule_sha256": schedule_hash, "config_sha256": sha256_file(config_path),
            },
            path,
        )
        artifacts[method] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    after = sha256_file(checkpoint)
    if after != before:
        raise AssertionError("V1 checkpoint changed during probe fit")
    manifest = {
        "study": cfg["study"], "fit_role": "clean_v6_train", "development_accessed": False,
        "official_accessed": False, "merger_updates": 0, "v1_inference_changed": False,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after, "train_cache": train_provenance,
        "source_range": cfg["source_ranges"]["train"], "fit_rows": len(selected),
        "fit_rows_per_corpus": int(cfg["probe"]["fit_rows_per_corpus"]),
        "fit_group_order_sha256": stable_hash(train_ids), "selection": selection,
        "tensor_identical_initialization": True, "initialization_sha256": initialization_hash,
        "identical_minibatch_schedule": True, "schedule_sha256": schedule_hash,
        "probe_protocol": cfg["probe"], "actual_device": str(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "final_training_metrics": finals, "probes": artifacts,
    }
    atomic_json(output / "PROBE_MANIFEST.json", manifest)
    return output / "PROBE_MANIFEST.json"


def _published_official_orders() -> dict[str, str]:
    published = ROOT / "results/official_v6/results.json"
    if sha256_file(published) != PUBLISHED_OFFICIAL_RESULTS_SHA256:
        raise ValueError("published official result artifact changed")
    result = json.loads(published.read_text())
    return {key: value["document_order_sha256"] for key, value in result["cache_provenance"]["corpora"].items()}


def prepare_official(config_path: Path, device_name: str) -> Path:
    """Regenerate a reduced safe-row cache only when the original cache is absent."""
    cfg = load_config(config_path)
    resolve_device(device_name)
    preferred = ROOT / cfg["official_cache_preferred"]
    if preferred.exists():
        raise FileExistsError("original official cache exists; evaluation must reuse it directly")
    destination = ROOT / cfg["official_cache_additive"]
    if destination.exists():
        raise FileExistsError("additive official-safe cache is create-only")
    expected_orders = _published_official_orders()
    if expected_orders != cfg["official_population"]["published_document_order_sha256"]:
        raise ValueError("config does not match published official document order")
    stage = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    stage.mkdir(parents=True)
    try:
        # Lazy imports keep fit/test/audit free of source-loader access.
        from recursive_geodesic_composition.data import FrozenTeacher, tokenize_text
        official_scripts = ROOT / "scripts" / "official_v6"
        sys.path.insert(0, str(official_scripts))
        from prepare_official_test_v6 import _stream_scan_window, exact_chunks, safe_prefix_length

        teacher = FrozenTeacher("Alibaba-NLP/gte-modernbert-base", "e7f32e3c00f91d699e8c43b53106206bcc72bb22", device_name)
        all_summary = {}
        start, end = cfg["source_ranges"]["official"]
        for corpus_cfg in cfg["corpora"]:
            corpus, rows, scanned = corpus_cfg["id"], [], 0
            for source_index, raw in _stream_scan_window(corpus_cfg, start, end):
                scanned += 1
                text = raw.get("text") or raw.get("content") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                token_ids = tokenize_text(teacher.tokenizer, text)
                prefix_count = safe_prefix_length(len(token_ids))
                if prefix_count is None:
                    continue
                chunks = exact_chunks(token_ids)
                full_leaves = teacher.embed(chunks, 16).float()
                safe_count = (prefix_count + 1023) // 1024
                safe_chunks = chunks[:safe_count]
                if sum(map(len, safe_chunks)) != prefix_count:
                    raise AssertionError("official safe-prefix chunk accounting changed")
                row = {
                    "id": f"{corpus}:{source_index}", "corpus": corpus,
                    "source_index": int(source_index), "selected_rank": len(rows),
                    "safe_prefix_token_count": int(prefix_count),
                    "leaves": full_leaves[:safe_count].cpu(),
                    "lengths": torch.tensor([len(chunk) for chunk in safe_chunks], dtype=torch.float32),
                }
                rows.append(row)
                if len(rows) == int(cfg["official_population"]["selected_per_corpus"]):
                    break
            ids = [row["id"] for row in rows]
            if len(rows) != 1000 or stable_hash(ids) != expected_orders[corpus]:
                raise ValueError(f"regenerated official population differs: {corpus}")
            exact = sum(len(row["leaves"]) == 4 for row in rows)
            if exact != int(cfg["official_population"]["expected_exact_k4_per_corpus"][corpus]):
                raise ValueError(f"official exact-K4 count differs: {corpus}")
            directory = stage / corpus
            directory.mkdir()
            path = directory / "safe.pt"
            torch.save(rows, path)
            all_summary[corpus] = {
                "rows": len(rows), "exact_k4_rows": exact, "scanned": scanned,
                "actual_scanned_range": [start, int(rows[-1]["source_index"]) + 1],
                "document_order_sha256": stable_hash(ids), "safe_tensor_sha256": sha256_file(path),
            }
        manifest = {
            "study": cfg["study"], "role": "official_test_safe_reconstruction",
            "source_range": cfg["source_ranges"]["official"], "selection": "published_official_v6_first_1000_safe",
            "population_identity_verified_against_published_results": True,
            "published_results_sha256": PUBLISHED_OFFICIAL_RESULTS_SHA256,
            "development_accessed": False, "fit_or_selection_on_official": False,
            "corpora": all_summary,
        }
        atomic_json(stage / "CACHE_MANIFEST.json", manifest)
        os.replace(stage, destination)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _load_official(cfg: dict) -> tuple[list[tuple[str, dict]], dict]:
    preferred, additive = ROOT / cfg["official_cache_preferred"], ROOT / cfg["official_cache_additive"]
    cache = preferred if preferred.exists() else additive
    if not cache.exists():
        raise FileNotFoundError("no official cache; run prepare-official")
    expected_orders = _published_official_orders()
    published = json.loads((ROOT / "results/official_v6/results.json").read_text())
    cache_manifest = json.loads((cache / "CACHE_MANIFEST.json").read_text())
    if cache == preferred:
        expected_manifest_hash = published["cache_provenance"]["manifest_sha256"]
        if sha256_file(cache / "CACHE_MANIFEST.json") != expected_manifest_hash:
            raise ValueError("original official cache manifest differs from published evaluation")
    elif (
        cache_manifest.get("role") != "official_test_safe_reconstruction"
        or cache_manifest.get("source_range") != cfg["source_ranges"]["official"]
        or cache_manifest.get("published_results_sha256") != PUBLISHED_OFFICIAL_RESULTS_SHA256
        or cache_manifest.get("fit_or_selection_on_official") is not False
    ):
        raise ValueError("additive official cache provenance changed")
    selected, provenance = [], {"path": str(cache.relative_to(ROOT)), "reused_original_cache": cache == preferred, "corpora": {}}
    for item in cfg["corpora"]:
        corpus = item["id"]
        safe_path = cache / corpus / "safe.pt"
        expected_tensor_hash = (
            published["cache_provenance"]["corpora"][corpus]["safe_tensor_sha256"]
            if cache == preferred else cache_manifest["corpora"][corpus]["safe_tensor_sha256"]
        )
        if sha256_file(safe_path) != expected_tensor_hash:
            raise ValueError(f"official safe tensor hash mismatch: {corpus}")
        rows = torch.load(safe_path, map_location="cpu", weights_only=False)
        ids = [str(row["id"]) for row in rows]
        if len(rows) != 1000 or stable_hash(ids) != expected_orders[corpus]:
            raise ValueError(f"official rows changed for {corpus}")
        exact = [row for row in rows if int(row["leaves"].shape[0]) == 4]
        expected = int(cfg["official_population"]["expected_exact_k4_per_corpus"][corpus])
        exact_hash = stable_hash([row["id"] for row in exact])
        if len(exact) != expected or exact_hash != cfg["official_population"]["expected_exact_k4_order_sha256"][corpus]:
            raise ValueError(f"official exact-K4 row identity changed for {corpus}")
        selected.extend((corpus, row) for row in exact)
        provenance["corpora"][corpus] = {
            "all_rows": len(rows), "exact_k4_rows": len(exact),
            "document_order_sha256": stable_hash(ids),
            "exact_k4_order_sha256": exact_hash,
            "source_index_min": min(int(row["source_index"]) for row in rows),
            "source_index_max": max(int(row["source_index"]) for row in rows),
        }
    if len(selected) != int(cfg["official_population"]["expected_exact_k4_total"]):
        raise ValueError("pooled official exact-K4 count changed")
    provenance["exact_k4_rows"] = len(selected)
    provenance["exact_k4_pooled_order_sha256"] = stable_hash([row["id"] for _, row in selected])
    if (cache / "CACHE_MANIFEST.json").exists():
        provenance["cache_manifest_sha256"] = sha256_file(cache / "CACHE_MANIFEST.json")
    return selected, provenance


def load_probes(cfg: dict, device: torch.device):
    manifest_path = ROOT / cfg["output_dir"] / "PROBE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    probes, identities = {}, {}
    for method in METHODS:
        entry = manifest["probes"][method]
        path = ROOT / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"probe hash changed: {method}")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("representation") != method or artifact.get("fit_role") != "clean_v6_train" or artifact.get("config_sha256") != sha256_file(CONFIG):
            raise ValueError("probe role/representation/config mismatch")
        if artifact.get("initialization_sha256") != manifest["initialization_sha256"] or artifact.get("schedule_sha256") != manifest["schedule_sha256"]:
            raise ValueError("probe matching identity mismatch")
        probe = ParentOnlyRetentionProbe(768, 4, 512)
        probe.load_state_dict(artifact["state_dict"], strict=True)
        probes[method] = probe.to(device).eval()
        identities[method] = entry
    return probes, identities, sha256_file(manifest_path)


def _scores(probe, parents, targets, device):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(parents), 512):
            predictions.append(probe(parents[start : start + 512].to(device)).cpu())
    prediction = torch.cat(predictions)
    return F.cosine_similarity(prediction, targets, dim=-1).mean(-1).numpy().astype(np.float64)


def paired_bootstrap(delta, reps: int, seed: int) -> list[float]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("paired bootstrap requires >=2 finite document deltas")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(reps), dtype=np.float64)
    for index in range(int(reps)):
        estimates[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def _scope(values, mask):
    picked = np.asarray(values)[mask]
    return {"rows": int(len(picked)), "mean_ordered_child_cosine": float(picked.mean())}


def evaluate(config_path: Path, device_name: str) -> Path:
    cfg = load_config(config_path)
    output = ROOT / cfg["output_dir"]
    result_path, scores_path = output / "RESULTS.json", output / "DOCUMENT_SCORES.json"
    if result_path.exists() or scores_path.exists():
        raise FileExistsError("additive official result namespace is create-only")
    checkpoint = ROOT / cfg["checkpoint"]
    before = sha256_file(checkpoint)
    if before != cfg["checkpoint_sha256"]:
        raise ValueError("frozen V1 checkpoint hash mismatch")
    selected, official = _load_official(cfg)
    train_manifest = json.loads((output / "PROBE_MANIFEST.json").read_text())
    # Disjoint pinned source-index windows make row overlap impossible; the
    # audit records the explicit interval intersection rather than pretending
    # raw TRAIN IDs were retained in the compact probe manifest.
    train_range = set(range(*cfg["source_ranges"]["train"]))
    official_range = set(range(*cfg["source_ranges"]["official"]))
    source_index_overlap = len(train_range & official_range)
    if train_manifest["source_range"] != cfg["source_ranges"]["train"] or source_index_overlap:
        raise ValueError("probe source role changed or overlaps official")
    official_ids = [str(row["id"]) for _, row in selected]
    if any(not cfg["source_ranges"]["official"][0] <= int(row["source_index"]) < cfg["source_ranges"]["official"][1] for _, row in selected):
        raise ValueError("official row escaped official range")
    device = resolve_device(device_name)
    model, _ = build_model(checkpoint, device)
    parents, targets = parent_features(model, selected, device)
    del model
    probes, probe_identities, probe_manifest_hash = load_probes(cfg, device)
    values = {method: _scores(probes[method], parents[method], targets, device) for method in METHODS}
    labels = np.asarray([corpus for corpus, _ in selected])
    records = []
    for index, (corpus, row) in enumerate(selected):
        records.append({
            "id": str(row["id"]), "corpus": corpus, "source_index": int(row["source_index"]),
            "score_v1": float(values["v1"][index]), "score_weighted": float(values["weighted"][index]),
            "score_mean": float(values["mean"][index]),
            "delta_v1_minus_weighted": float(values["v1"][index] - values["weighted"][index]),
            "delta_v1_minus_mean": float(values["v1"][index] - values["mean"][index]),
        })
    atomic_json(scores_path, {"study": cfg["study"], "rows": records})
    scopes = {}
    names = ("pooled", *(item["id"] for item in cfg["corpora"]))
    for offset, scope in enumerate(names):
        mask = np.ones(len(selected), dtype=bool) if scope == "pooled" else labels == scope
        delta_weighted = values["v1"][mask] - values["weighted"][mask]
        delta_mean = values["v1"][mask] - values["mean"][mask]
        seed = int(cfg["bootstrap"]["seed"]) + 10 * offset
        scopes[scope] = {
            "rows": int(mask.sum()),
            "v1": _scope(values["v1"], mask), "weighted": _scope(values["weighted"], mask),
            "mean": _scope(values["mean"], mask),
            "v1_minus_weighted": {"mean": float(delta_weighted.mean()), "paired_document_bootstrap_ci95": paired_bootstrap(delta_weighted, 2000, seed)},
            "v1_minus_mean": {"mean": float(delta_mean.mean()), "paired_document_bootstrap_ci95": paired_bootstrap(delta_mean, 2000, seed + 1)},
        }
    after = sha256_file(checkpoint)
    if after != before:
        raise AssertionError("V1 checkpoint changed during official evaluation")
    result = {
        "study": cfg["study"], "primary_v1_seed": cfg["primary_v1_seed"],
        "protocol": "matched_frozen_parent_only_probes_exact_official_safe_k4",
        "representations": {
            "v1": "frozen V1 K4 merger output",
            "weighted": "normalize(sum_i (n_i/sum_j n_j) * normalize(x_i))",
            "mean": "normalize(sum_i normalize(x_i))",
        },
        "target": "ordered tuple of four normalized child embeddings",
        "metric": "per-document mean of four ordered-child cosines",
        "fit_role": "clean_v6_train", "development_fitting_tuning_selection_updates": 0,
        "official_fitting_tuning_selection_updates": 0, "official_population": official,
        "official_exact_k4_rows": len(selected), "official_document_order_sha256": stable_hash(official_ids),
        "train_official_source_index_overlap": source_index_overlap, "range_overlap": bool(source_index_overlap),
        "checkpoint": cfg["checkpoint"], "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
        "probe_manifest_sha256": probe_manifest_hash, "probes": probe_identities,
        "tensor_identical_probe_initialization": True, "identical_probe_minibatch_schedule": True,
        "bootstrap": {"unit": "source_document", "pooled_resampling": "paper_convention_unstratified_paired_documents", **cfg["bootstrap"]},
        "actual_device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "scopes": scopes,
        "document_scores": str(scores_path.relative_to(ROOT)), "document_scores_sha256": sha256_file(scores_path),
    }
    atomic_json(result_path, result)
    return result_path


def audit(config_path: Path) -> Path:
    cfg = load_config(config_path)
    output = ROOT / cfg["output_dir"]
    result = json.loads((output / "RESULTS.json").read_text())
    probe = json.loads((output / "PROBE_MANIFEST.json").read_text())
    scores_path = output / "DOCUMENT_SCORES.json"
    scores = json.loads(scores_path.read_text())["rows"]
    labels = np.asarray([row["corpus"] for row in scores])
    arrays = {method: np.asarray([row[f"score_{method}"] for row in scores], dtype=np.float64) for method in METHODS}
    rows_consistent = all(
        np.isclose(row["delta_v1_minus_weighted"], row["score_v1"] - row["score_weighted"], rtol=0, atol=1e-15)
        and np.isclose(row["delta_v1_minus_mean"], row["score_v1"] - row["score_mean"], rtol=0, atol=1e-15)
        for row in scores
    )
    summaries_consistent = True
    for offset, scope in enumerate(("pooled", *(item["id"] for item in cfg["corpora"]))):
        mask = np.ones(len(scores), dtype=bool) if scope == "pooled" else labels == scope
        recorded = result["scopes"][scope]
        summaries_consistent &= recorded["rows"] == int(mask.sum())
        summaries_consistent &= all(
            np.isclose(recorded[method]["mean_ordered_child_cosine"], arrays[method][mask].mean(), rtol=0, atol=1e-15)
            for method in METHODS
        )
        for comparison, baseline, seed_add in (("v1_minus_weighted", "weighted", 0), ("v1_minus_mean", "mean", 1)):
            delta = arrays["v1"][mask] - arrays[baseline][mask]
            summaries_consistent &= np.isclose(recorded[comparison]["mean"], delta.mean(), rtol=0, atol=1e-15)
            expected_ci = paired_bootstrap(delta, 2000, int(cfg["bootstrap"]["seed"]) + 10 * offset + seed_add)
            summaries_consistent &= np.allclose(recorded[comparison]["paired_document_bootstrap_ci95"], expected_ci, rtol=0, atol=1e-15)
    checks = {
        "v1_checkpoint_hash_unchanged": result["checkpoint_sha256_before"] == result["checkpoint_sha256_after"] == cfg["checkpoint_sha256"] == sha256_file(ROOT / cfg["checkpoint"]),
        "published_official_results_unchanged": sha256_file(ROOT / "results/official_v6/results.json") == PUBLISHED_OFFICIAL_RESULTS_SHA256,
        "official_rows_match_published_population": all(value["document_order_sha256"] == cfg["official_population"]["published_document_order_sha256"][corpus] and value["exact_k4_order_sha256"] == cfg["official_population"]["expected_exact_k4_order_sha256"][corpus] and stable_hash([row["id"] for row in scores if row["corpus"] == corpus]) == cfg["official_population"]["expected_exact_k4_order_sha256"][corpus] for corpus, value in result["official_population"]["corpora"].items()),
        "official_exact_k4_count": result["official_exact_k4_rows"] == len(scores) == cfg["official_population"]["expected_exact_k4_total"],
        "train_only_probe_fit": probe["fit_role"] == "clean_v6_train" and probe["development_accessed"] is False and probe["official_accessed"] is False,
        "no_train_dev_official_range_overlap": cfg["source_ranges"] == {"train": [140000, 152000], "development": [152000, 161000], "buffer": [161000, 162000], "official": [162000, 192000]} and result["train_official_source_index_overlap"] == 0 and result["range_overlap"] is False,
        "all_official_rows_in_range": all(162000 <= int(row["source_index"]) < 192000 for row in scores),
        "no_duplicate_official_documents": len({row["id"] for row in scores}) == len(scores),
        "document_score_identity_and_hash_match": stable_hash([row["id"] for row in scores]) == result["official_document_order_sha256"] and sha256_file(scores_path) == result["document_scores_sha256"],
        "probe_manifest_hash_match": sha256_file(output / "PROBE_MANIFEST.json") == result["probe_manifest_sha256"],
        "document_deltas_recomputed": bool(rows_consistent),
        "means_and_bootstrap_cis_recomputed": bool(summaries_consistent),
        "zero_official_updates": result["official_fitting_tuning_selection_updates"] == 0,
        "matched_initialization_and_schedule": result["tensor_identical_probe_initialization"] is True and result["identical_probe_minibatch_schedule"] is True,
        "all_probe_hashes_match": all(sha256_file(ROOT / entry["path"]) == entry["sha256"] for entry in probe["probes"].values()),
        "finite_scores_and_deltas": all(np.isfinite([value for key, value in row.items() if key.startswith("score_") or key.startswith("delta_")]).all() for row in scores),
    }
    summary = {
        "study": cfg["study"], "passed": all(checks.values()), "checks": checks,
        "source_ranges": cfg["source_ranges"], "row_counts": {"train_probe_fit": probe["fit_rows"], "official_exact_k4": len(scores)},
        "hashes": {
            "config": sha256_file(config_path), "implementation": sha256_file(Path(__file__)),
            "v1_checkpoint": sha256_file(ROOT / cfg["checkpoint"]),
            "published_official_results": sha256_file(ROOT / "results/official_v6/results.json"),
            "probe_manifest": sha256_file(output / "PROBE_MANIFEST.json"),
            "results": sha256_file(output / "RESULTS.json"), "document_scores": sha256_file(output / "DOCUMENT_SCORES.json"),
        },
    }
    atomic_json(output / "AUDIT.json", summary)
    if not summary["passed"]:
        raise AssertionError("V1 K4 pooling audit failed")
    return output / "AUDIT.json"


def report(config_path: Path) -> Path:
    cfg = load_config(config_path)
    output = ROOT / cfg["output_dir"]
    result, audit_result = json.loads((output / "RESULTS.json").read_text()), json.loads((output / "AUDIT.json").read_text())
    pooled = result["scopes"]["pooled"]
    def verdict(comparison):
        low, high = comparison["paired_document_bootstrap_ci95"]
        if low > 0: return "V1 is higher under this matched protocol."
        if high < 0: return "Pooling is higher under this matched protocol."
        return "The interval overlaps zero; this experiment does not establish superiority."
    lines = [
        "# V1 exact-K4 matched pooling retention", "",
        "Frozen V1 seed/checkpoint: `424242` / `989f9bc496bea484974679cd181e43836caf3283ffdc8e66507046945e0902de`.",
        "All three separately fitted probes use only the parent, the same architecture, tensor-identical initialization, and the same 400-update TRAIN-only minibatch schedule. The evaluation population is the published official-v6 safe exact-K4 subset.", "",
        "| Method | Mean ordered-child cosine |", "|---|---:|",
        f"| V1 learned parent | {pooled['v1']['mean_ordered_child_cosine']:.9f} |",
        f"| Token-mass-weighted normalized mean | {pooled['weighted']['mean_ordered_child_cosine']:.9f} |",
        f"| Unweighted normalized mean | {pooled['mean']['mean_ordered_child_cosine']:.9f} |", "",
        "## Per-corpus means", "",
        "| Corpus | Rows | V1 | Weighted | Unweighted |", "|---|---:|---:|---:|---:|",
        *[
            f"| {scope} | {result['scopes'][scope]['rows']} | {result['scopes'][scope]['v1']['mean_ordered_child_cosine']:.9f} | {result['scopes'][scope]['weighted']['mean_ordered_child_cosine']:.9f} | {result['scopes'][scope]['mean']['mean_ordered_child_cosine']:.9f} |"
            for scope in (item["id"] for item in cfg["corpora"])
        ], "",
        "| Paired difference | Mean | 95% document bootstrap CI |", "|---|---:|---:|",
        f"| V1 - weighted | {pooled['v1_minus_weighted']['mean']:.9f} | [{pooled['v1_minus_weighted']['paired_document_bootstrap_ci95'][0]:.9f}, {pooled['v1_minus_weighted']['paired_document_bootstrap_ci95'][1]:.9f}] |",
        f"| V1 - unweighted | {pooled['v1_minus_mean']['mean']:.9f} | [{pooled['v1_minus_mean']['paired_document_bootstrap_ci95'][0]:.9f}, {pooled['v1_minus_mean']['paired_document_bootstrap_ci95'][1]:.9f}] |", "",
        f"Weighted comparison: {verdict(pooled['v1_minus_weighted'])}",
        f"Unweighted comparison: {verdict(pooled['v1_minus_mean'])}", "",
        "Under this specified nonlinear matched parent-only probe protocol, both pooling baselines preserve more probe-recoverable ordered child information than V1. The required legacy probe is residual around its parent input (and initializes to the parent), so this result is specific to that architecture and may favor mean-like representations. The experiment does not establish lossless encoding, preserved mutual information, architecture-independent recoverability, universal downstream superiority, or semantic superiority.", "",
        f"Official exact-K4 documents: **{result['official_exact_k4_rows']}**. Audit passed: **{audit_result['passed']}**.", "",
        "## Reproduction", "", "```bash",
        "# Preserve the committed machine-readable evidence before a create-only reproduction:",
        "mv results/v1_k4_matched_pooling results/v1_k4_matched_pooling.published",
        "# Materialize the release checkpoint:",
        "# Obtain the exact final checkpoint through the anonymous submission discussion.",
        "python supplement.py materialize /path/to/weights --role final_clean_v6_checkpoint --mode hardlink",
        "PYTHONPATH=src python scripts/final_v6/prepare_train.py --config configs/final_v6/seed424242.yaml --device cuda",
        "PYTHONPATH=src python scripts/v1_k4_pooling.py prepare-official --device cuda",
        "PYTHONPATH=src python scripts/v1_k4_pooling.py fit --device cuda",
        "PYTHONPATH=src python scripts/v1_k4_pooling.py evaluate --device cuda",
        "PYTHONPATH=src python scripts/v1_k4_pooling.py audit",
        "PYTHONPATH=src python scripts/v1_k4_pooling.py report", "```", "",
        "If the original `cache_kary4_strict_autoencoder_official_test_v6` is present, omit `prepare-official`; evaluation reuses it preferentially.",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    hashes = {}
    for artifact in ("PROBE_MANIFEST.json", "RESULTS.json", "DOCUMENT_SCORES.json", "AUDIT.json", "REPORT.md"):
        hashes[artifact] = sha256_file(output / artifact)
    for method in METHODS:
        name = f"probes/{method}_parent_only.pt"
        hashes[name] = sha256_file(output / name)
    atomic_json(output / "ARTIFACT_HASHES.json", hashes)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-official", "fit", "evaluate", "audit", "report"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    commands = {
        "prepare-official": lambda: prepare_official(args.config, args.device),
        "fit": lambda: fit(args.config, args.device),
        "evaluate": lambda: evaluate(args.config, args.device),
        "audit": lambda: audit(args.config),
        "report": lambda: report(args.config),
    }
    print(commands[args.command]())


if __name__ == "__main__":
    main()
