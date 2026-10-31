from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts/final_v6"
sys.path.insert(0, str(STAGE))

from recursive_geodesic_composition.architecture import GeodesicMerge
from protocol import (DEVELOPMENT_SCAN, FUTURE_PROTECTED, RESERVE_BUFFER, SEEDS,
                      TRAIN_SCAN, disjoint, load_config, validate_config)
from recursive_geodesic_composition.train import _load_encoder_initialization, _load_training_roles


def test_all_seed_configs_are_exact_and_unique():
    outputs = set()
    for seed in SEEDS:
        cfg = load_config(ROOT / f"configs/final_v6/seed{seed}.yaml")
        assert cfg["seed"] == seed
        assert cfg["training_without_validation"] is True
        assert cfg["matched_probe"]["fit_rows_per_corpus"] == 80
        assert cfg["matched_probe"]["floor_rows_per_corpus"] == 80
        outputs.add(cfg["output_dir"])
    assert len(outputs) == len(SEEDS)


def test_ranges_are_exact_disjoint_and_adjacent():
    ranges = (TRAIN_SCAN, DEVELOPMENT_SCAN, RESERVE_BUFFER, FUTURE_PROTECTED)
    assert all(disjoint(a, b) for i, a in enumerate(ranges) for b in ranges[i + 1:])
    assert TRAIN_SCAN[1] == DEVELOPMENT_SCAN[0]
    assert DEVELOPMENT_SCAN[1] == RESERVE_BUFFER[0]
    assert RESERVE_BUFFER[1] == FUTURE_PROTECTED[0]


def test_protocol_rejects_boundary_and_recipe_mutation():
    config_path = ROOT / "configs/final_v6/seed424242.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["corpora"][0]["train_start"] += 1
    with pytest.raises(ValueError): validate_config(cfg)
    cfg = yaml.safe_load(config_path.read_text())
    cfg["steps"] -= 1
    with pytest.raises(ValueError): validate_config(cfg)


def test_train_only_loader_never_substitutes_validation(tmp_path):
    corpora = []
    for name in ("a", "b"):
        directory = tmp_path / name; directory.mkdir()
        torch.save([{"x": name}], directory / "train.pt")
        corpora.append({"id": name, "weight": .5})
    rows, validation, multicorpus = _load_training_roles({"cache_dir": str(tmp_path), "corpora": corpora,
                                                          "training_without_validation": True})
    assert len(rows) == 2 and validation == {} and multicorpus is not None
    torch.save([{"x": "forbidden"}], tmp_path / "a" / "validation.pt")
    with pytest.raises(ValueError):
        _load_training_roles({"cache_dir": str(tmp_path), "corpora": corpora,
                              "training_without_validation": True})


def _model():
    return GeodesicMerge(d=8, hidden=16, layers=1, heads=2, max_angle=.5, decoder=True,
                         decoder_layers=1, decoder_length_mode="predicted", importance_mode="semantic",
                         decoder_mode="strict", geodesic_parameterization="tangent_expmap", arity=4)


def test_autoencoder_source_loads_only_encoder(tmp_path):
    torch.manual_seed(1); source = _model()
    checkpoint = tmp_path / "source.pt"
    torch.save({"variant": "autoencoder", "selected_step": 400, "state_dict": source.state_dict(),
                "architecture": source.architecture_metadata(), "config": {}}, checkpoint)
    torch.manual_seed(2); target = _model()
    decoder_before = {k: v.clone() for k, v in target.state_dict().items() if k.startswith("strict_decoder.")}
    info = _load_encoder_initialization(target, checkpoint, allow_autoencoder_source=True)
    assert info["continued_autoencoder_retraining"] is True
    assert info["initialization_ignored_decoder_key_count"] > 0
    for name, value in decoder_before.items(): assert torch.equal(value, target.state_dict()[name])
    assert torch.equal(source.state_dict()["input.0.weight"], target.state_dict()["input.0.weight"])
    with pytest.raises(ValueError): _load_encoder_initialization(_model(), checkpoint)


def test_development_preparer_never_requests_reserve_or_future():
    text = (STAGE / "prepare_development.py").read_text()
    assert "dataset.skip(DEVELOPMENT_SCAN[0]).take(maximum)" in text
    assert "if offset >= maximum" in text
    assert "RESERVE_BUFFER" in text and "FUTURE_PROTECTED" in text
