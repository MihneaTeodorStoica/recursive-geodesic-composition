from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from recursive_geodesic_composition.evaluate_v3_cls import (
    CORPORA,
    evaluate_full_scopes,
    evaluate_safe_scopes,
    infer_full,
    infer_safe,
    load_checkpoint,
    load_two_arm_cache,
    sha256_file,
    validate_protocol,
)
from recursive_geodesic_composition.architecture import GeodesicMerge

ROOT = Path(__file__).resolve().parents[1]


class SyntheticK4(torch.nn.Module):
    """Small deterministic merger exposing the production telemetry contract."""
    def __init__(self, dimension=8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.d = dimension
        self.arity = 4
        self.max_angle = 0.5
        self.importance_beta = 1.0
        self._last_merge_aux = None

    def merge(self, children, masses):
        active = masses > 0
        weights = masses / masses.sum(-1, keepdim=True)
        values = F.normalize((children * weights[..., None]).sum(1) + 0.01, dim=-1)
        batch = len(children)
        self._last_merge_aux = {
            "active_mask": active,
            "corrections": torch.zeros_like(masses),
            "angle": torch.full((batch,), 0.1, device=children.device),
            "geodesic_saturated": torch.zeros(batch, dtype=torch.bool, device=children.device),
        }
        return values


def protocol():
    return yaml.safe_load((ROOT / "configs/v3_cls/evaluator_two_arm.yaml").read_text())


def vectors(count, dimension=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)


def safe_by(count=12):
    result = {}
    for ci, corpus in enumerate(CORPORA):
        rows = []
        for i in range(count):
            leaves = vectors(2 + i % 3, seed=1000 * ci + i)
            lengths = torch.tensor([1024] * (len(leaves) - 1) + [1 + i % 1024], dtype=torch.float32)
            target = F.normalize(leaves.mean(0) + 0.02 * (ci + 1), dim=0)
            rows.append({"id": f"{corpus}:{i}", "source_index": 308000 + i,
                         "leaves": leaves, "lengths": lengths, "teacher_target": target,
                         "teacher_pooling": "cls"})
        result[corpus] = rows
    return result


def full_by(count=3):
    result = {}
    for ci, corpus in enumerate(CORPORA):
        items = []
        for i in range(count):
            leaf_count = (2, 5, 17)[i % 3]
            lengths = torch.tensor([1024] * (leaf_count - 1) + [17], dtype=torch.float32)
            shard = {"id": f"{corpus}:{i}", "source_index": 308000 + i,
                     "leaves": vectors(leaf_count, seed=9000 + 100 * ci + i), "lengths": lengths,
                     "token_count": int(lengths.sum()), "truncated": False,
                     "token_cap": None, "leaf_cap": None}
            items.append({"id": shard["id"], "source_index": shard["source_index"],
                          "token_count": shard["token_count"], "_loaded_shard": shard,
                          "_source_range": (308000, 317000)})
        result[corpus] = items
    return result


def test_frozen_protocol_and_no_full_teacher_metrics():
    cfg = protocol()
    validate_protocol(cfg)
    assert cfg["safe_arm"]["minimum_tokens"] == 1025
    assert cfg["safe_arm"]["maximum_tokens"] == 4096
    assert cfg["bootstrap_replicates"] == 400
    assert cfg["full_arm"]["teacher_target"] is False
    assert cfg["full_arm"]["teacher_metrics"] is False
    assert cfg["full_arm"]["attacks"] is False
    changed = copy.deepcopy(cfg); changed["full_arm"]["depth_cap"] = 4
    with pytest.raises(ValueError, match="full arm"):
        validate_protocol(changed)


def test_synthetic_safe_is_direct_with_telemetry_attacks_bootstrap_and_corpora():
    model = SyntheticK4()
    data = infer_safe(model, safe_by(), torch.device("cpu"), batch_size=5)
    assert data["roots"].shape == (len(CORPORA) * 12, 8)
    assert data["telemetry"]["depth"] == {"minimum": 1, "mean": 1.0, "maximum": 1}
    assert data["telemetry"]["geodesic_saturation_count"] == 0
    cfg = protocol(); cfg["bootstrap_replicates"] = 20
    attack_cfg = yaml.safe_load((ROOT / cfg["attack_config"]).read_text())
    scopes = evaluate_safe_scopes(data, cfg, attack_cfg)
    assert tuple(scopes) == ("pooled", *CORPORA)
    for value in scopes.values():
        assert np.isfinite(value["teacher_cosine"])
        assert len(value["teacher_cosine_ci95_bootstrap"]) == 2
        assert set(value["attacks"]["attacks"]) == {"standard_scaled_logistic", "standard_scaled_rbf_svm"}
        assert np.isfinite(value["health"]["effective_rank"])
        assert value["health"]["finite_fraction"] == 1.0


def test_synthetic_full_recurses_all_leaves_and_emits_health_only():
    model = SyntheticK4()
    data = infer_full(model, full_by(), torch.device("cpu"), batch_size=2)
    assert data["roots"].shape == (len(CORPORA) * 3, 8)
    assert data["telemetry"]["leaves"] == len(CORPORA) * (2 + 5 + 17)
    assert data["telemetry"]["depth"]["maximum"] == 3
    result = evaluate_full_scopes(data, protocol())
    assert tuple(result) == ("pooled", *CORPORA)
    serialized = json.dumps(result)
    assert "teacher" not in serialized and "attack" not in serialized and "cosine" not in serialized
    assert all(value["health"]["finite_fraction"] == 1.0 for value in result.values())


def _write_cache(tmp_path):
    cache = tmp_path / "artifacts/v3_cls/cache_development_cls"
    artifacts = {}
    safe_rows = safe_by(count=2)
    for corpus in CORPORA:
        directory = cache / corpus; directory.mkdir(parents=True)
        torch.save(safe_rows[corpus], directory / "safe.pt")
        artifacts[f"{corpus}/safe.pt"] = {"sha256": sha256_file(directory / "safe.pt")}
        documents = []
        for i, row in enumerate(safe_rows[corpus]):
            lengths = torch.tensor([1024, 1024, 13], dtype=torch.float32)
            shard = {"id": row["id"], "source_index": row["source_index"], "leaves": vectors(3, seed=700+i),
                     "lengths": lengths, "token_count": int(lengths.sum()), "truncated": False,
                     "token_cap": None, "leaf_cap": None}
            relative = f"{corpus}/full_{i}.pt"
            torch.save(shard, cache / relative)
            artifacts[relative] = {"sha256": sha256_file(cache / relative)}
            documents.append({"id": row["id"], "source_index": row["source_index"],
                              "token_count": shard["token_count"], "shard": relative})
        (directory / "full_index.json").write_text(json.dumps({"documents": documents}))
        artifacts[f"{corpus}/full_index.json"] = {"sha256": sha256_file(directory / "full_index.json")}
    cfg = protocol()
    manifest = {"schema": "v3_cls_two_arm_cache_v1", "study": "v3_cls", "role": "development",
                "source_range": [308000, 317000], "teacher_provenance": {
                    "model": cfg["model_name"], "revision": cfg["model_revision"], "pooling": "cls"},
                "safe_arm": cfg["safe_arm"], "full_arm": cfg["full_arm"], "artifacts": artifacts}
    manifest_path = cache / "CACHE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    (cache / "COMPLETE").write_text(f"{sha256_file(manifest_path)}  CACHE_MANIFEST.json\n")
    spec = {"role": "development", "cache_manifest": str(manifest_path.relative_to(tmp_path)),
            "cache_manifest_sha256": sha256_file(manifest_path)}
    return cache, manifest_path, spec


def _write_checkpoint(tmp_path, *, v3=True):
    cache = tmp_path / "artifacts/v3_cls/cache_train_cls"
    cache.mkdir(parents=True)
    manifest_path = cache / "CACHE_MANIFEST.json"
    manifest_path.write_text(json.dumps({"study": "v3_cls", "role": "train", "source_range": [296000, 308000],
        "teacher_provenance": {"teacher_id": protocol()["model_name"],
                               "teacher_revision": protocol()["model_revision"], "teacher_pooling": "cls"}}))
    cfg = {"v3_cls_experiment": v3, "teacher_pooling": "cls", "model_name": protocol()["model_name"],
           "model_revision": protocol()["model_revision"], "teacher_trust_remote_code": True,
           "cache_dir": str(cache), "output_dir": str(tmp_path / "results/v3_cls/seed7"),
           "arity": 4, "encoder_hidden": 16, "encoder_layers": 1, "heads": 2,
           "max_geodesic_angle": .5, "decoder_layers": 1, "decoder_length_mode": "predicted",
           "importance_mode": "semantic", "importance_beta": 1.0, "decoder_mode": "strict",
           "geodesic_parameterization": "tangent_expmap", "checkpoint_selection": "fixed_final_train_only",
           "steps": 3, "initialize_encoder_from_autoencoder_checkpoint": False}
    model = GeodesicMerge(d=8, hidden=16, layers=1, heads=2, max_angle=.5, decoder=True,
                          decoder_layers=1, decoder_length_mode="predicted", importance_mode="semantic",
                          decoder_mode="strict", geodesic_parameterization="tangent_expmap", arity=4)
    checkpoint_path = tmp_path / "results/v3_cls/seed7/autoencoder_seed7.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"config": cfg, "seed": 7, "variant": "autoencoder", "selected_step": 3,
                "state_dict": model.state_dict()}, checkpoint_path)
    return {"checkpoint": str(checkpoint_path.relative_to(tmp_path)), "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_cache_manifest": str(manifest_path.relative_to(tmp_path)),
            "training_cache_manifest_sha256": sha256_file(manifest_path), "seed": 7}


def test_checkpoint_guard_accepts_only_hash_pinned_fresh_v3_cls(tmp_path):
    spec = _write_checkpoint(tmp_path)
    model, checkpoint, provenance = load_checkpoint(tmp_path, spec, torch.device("cpu"))
    assert model.arity == 4 and checkpoint["selected_step"] == 3 and provenance["fresh_v3_cls"] is True
    bad = dict(spec); bad["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(tmp_path, bad, torch.device("cpu"))


def test_checkpoint_guard_rejects_non_v3_lineage(tmp_path):
    spec = _write_checkpoint(tmp_path, v3=False)
    with pytest.raises(ValueError, match="does not declare"):
        load_checkpoint(tmp_path, spec, torch.device("cpu"))


def test_synthetic_cache_provenance_and_same_arm_order(tmp_path):
    _, _, spec = _write_cache(tmp_path)
    safe, full, provenance = load_two_arm_cache(tmp_path, spec, protocol())
    assert provenance["teacher_pooling"] == "cls"
    assert provenance["same_documents_both_arms"] is True
    assert all([row["id"] for row in safe[c]] == [item["id"] for item in full[c]] for c in CORPORA)


def test_cache_rejects_full_root_teacher_target(tmp_path):
    cache, manifest_path, spec = _write_cache(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    shard_rel = json.loads((cache / CORPORA[0] / "full_index.json").read_text())["documents"][0]["shard"]
    shard_path = cache / shard_rel
    shard = torch.load(shard_path, weights_only=False); shard["teacher_target"] = vectors(1)[0]
    torch.save(shard, shard_path)
    manifest["artifacts"][shard_rel]["sha256"] = sha256_file(shard_path)
    manifest_path.write_text(json.dumps(manifest)); spec["cache_manifest_sha256"] = sha256_file(manifest_path)
    (cache / "COMPLETE").write_text(f"{sha256_file(manifest_path)}  CACHE_MANIFEST.json\n")
    _, full, _ = load_two_arm_cache(tmp_path, spec, protocol())
    with pytest.raises(ValueError, match="forbidden teacher target"):
        infer_full(SyntheticK4(), full, torch.device("cpu"), batch_size=2)


def test_nonfinite_safe_tensor_fails_before_metrics():
    rows = safe_by(count=2)
    rows[CORPORA[0]][0]["leaves"][0, 0] = float("nan")
    # Cache validation is the required pre-inference finite barrier.
    from recursive_geodesic_composition.evaluate_v3_cls import _validate_safe_row
    with pytest.raises(FloatingPointError, match="non-finite"):
        _validate_safe_row(rows[CORPORA[0]][0], "development", (308000, 317000))
