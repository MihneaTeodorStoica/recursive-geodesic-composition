from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("safe_baselines_evaluator", ROOT / "scripts/evaluate_safe_baselines.py")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_padded_batches_preserve_document_identity_and_arity_grouping():
    arities = [4, 2, 3, 2, 4]
    rows = [{"leaves": F.normalize(torch.randn(k, 768), dim=-1),
             "lengths": torch.tensor([1024.] * (k - 1) + [17.])} for k in arities]
    batches, indices = EVALUATOR.padded_batches(rows, torch.device("cpu"), 2)
    seen = []
    for (children, lengths), selection in zip(batches, indices):
        assert len(children) <= 2
        assert len(set(arities[index] for index in selection)) == 1
        for local, original in enumerate(selection):
            count = arities[original]
            assert torch.equal(children[local, :count], rows[original]["leaves"])
            assert torch.equal(lengths[local, :count], rows[original]["lengths"])
            assert not lengths[local, count:].any()
            seen.append(original)
    assert sorted(seen) == list(range(len(rows)))


def test_attack_results_match_original_split_and_resume_without_refitting(tmp_path, monkeypatch):
    monkeypatch.setattr(EVALUATOR, "ROOT", tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "output/INFERENCE_MANIFEST.json").write_text("{}")
    (tmp_path / "attacks.yaml").write_text("test_size: 0.5\nsplit_seed: 1729\n")
    cfg = {"output_dir": "output", "attack_config": "attacks.yaml", "config_sha256": "configuration",
           "attack_config_sha256": "recipe", "bootstrap": {"reps": 2000, "baseline_attack_seed": 91}}
    groups = np.array([f"document:{index}" for index in range(20)])
    rng = np.random.default_rng(21)
    real, fake = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
    _, y, paired, train, held = EVALUATOR.grouped_attack_split(real, fake, groups, {"test_size": .5, "split_seed": 1729})
    metrics = {"train_group_sha256": EVALUATOR.stable_hash(sorted(np.unique(paired[train]).tolist())),
               "holdout_group_sha256": EVALUATOR.stable_hash(sorted(np.unique(paired[held]).tolist()))}
    published = {"seeds": [{"safe_arm": {"scopes": {"pooled": {"attacks": metrics}}}}]}
    calls = []

    def fitted(*args):
        calls.append(1)
        return metrics, {"example": {"y": y[held], "groups": paired[held], "scores": np.arange(len(held)) / len(held)}}

    monkeypatch.setattr(EVALUATOR, "evaluate_fixed_attacks", fitted)
    result = EVALUATOR.attack_result("unweighted_mean", "pooled", real, fake, groups, cfg, {}, published, 0)
    resumed = EVALUATOR.attack_result("unweighted_mean", "pooled", real, fake, groups, cfg, {}, published, 0)
    assert len(calls) == 1
    assert result == resumed
    assert not set(result["classifier_train_groups"]) & set(result["classifier_holdout_groups"])
    assert len(result["bundles"]["example"]["scores"]) == 20
    # A changed inference artifact invalidates the cached fitted scores.
    (tmp_path / "output/INFERENCE_MANIFEST.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="incompatible"):
        EVALUATOR.attack_result("unweighted_mean", "pooled", real, fake, groups, cfg, {}, published, 0)


def test_teacher_parameter_precision_is_distinguished_from_storage():
    cfg = yaml.safe_load((ROOT / "configs/safe_baselines_v1.yaml").read_text())
    assert cfg["preparation"]["teacher_load_dtype"] == "model_default_as_in_original_protocol"
    assert cfg["preparation"]["teacher_parameter_dtype"] == "float16"
    assert cfg["preparation"]["embedding_storage_dtype"] == "float32"
    assert cfg["inference"]["dtype"] == "float32"
