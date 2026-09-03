import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v1_k4_pooling", ROOT / "scripts/v1_k4_pooling.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_protocol_and_roles():
    cfg = MODULE.load_config(ROOT / "configs/v1_k4_pooling.yaml")
    assert cfg["source_ranges"] == {
        "train": [140000, 152000],
        "development": [152000, 161000],
        "buffer": [161000, 162000],
        "official": [162000, 192000],
    }
    assert cfg["official_population"]["expected_exact_k4_total"] == 606
    assert cfg["official_updates"] == 0


def test_pooling_parents_match_definitions_and_normalize_children():
    children = torch.tensor(
        [[[2.0, 0.0], [0.0, 3.0], [-4.0, 0.0], [0.0, -5.0]]]
    )
    masses = torch.tensor([[1.0, 2.0, 3.0, 8.0]])
    weighted, mean = MODULE.pooling_parents(children, masses)
    x = F.normalize(children, dim=-1)
    q = masses / masses.sum(-1, keepdim=True)
    assert torch.allclose(weighted, F.normalize((q[..., None] * x).sum(1), dim=-1))
    assert torch.allclose(mean, F.normalize(x.sum(1), dim=-1))
    assert torch.allclose(weighted.norm(dim=-1), torch.ones(1))


def test_pooling_rejects_nonpositive_masses():
    with pytest.raises(ValueError):
        MODULE.pooling_parents(torch.randn(1, 4, 3), torch.tensor([[1.0, 1.0, 0.0, 1.0]]))


def test_paired_bootstrap_is_deterministic_and_document_level():
    delta = np.array([0.1, 0.2, -0.1, 0.0])
    first = MODULE.paired_bootstrap(delta, 2000, 17)
    second = MODULE.paired_bootstrap(delta, 2000, 17)
    assert first == second
    assert first[0] <= delta.mean() <= first[1]


def test_train_selection_is_exact_k4_train_only_and_deterministic():
    cfg = MODULE.load_config(ROOT / "configs/v1_k4_pooling.yaml")
    cfg["probe"]["fit_rows_per_corpus"] = 2
    rows_by = {}
    for item in cfg["corpora"]:
        corpus = item["id"]
        rows_by[corpus] = [
            {"id": f"{corpus}:{i}", "segmentation": "train", "leaves": torch.ones(4, 768)}
            for i in range(4)
        ] + [{"id": "wrong-role", "segmentation": "validation", "leaves": torch.ones(4, 768)}]
    first, audit_first = MODULE.select_train_rows(rows_by, cfg)
    second, audit_second = MODULE.select_train_rows(rows_by, cfg)
    assert [row["id"] for _, row in first] == [row["id"] for _, row in second]
    assert audit_first == audit_second
    assert len(first) == 8
    assert all(row["segmentation"] == "train" and len(row["leaves"]) == 4 for _, row in first)


def test_parent_only_probe_has_only_parent_argument():
    probe = MODULE.ParentOnlyRetentionProbe(8, 4, 16)
    parent = torch.randn(3, 8)
    output = probe(parent)
    assert output.shape == (3, 4, 8)
    assert torch.allclose(output.norm(dim=-1), torch.ones(3, 4), atol=1e-6)
