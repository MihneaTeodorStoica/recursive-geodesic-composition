from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from recursive_geodesic_composition import GeodesicMerge, recursive_reduce

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_primary_result_and_publication_figure_hashes():
    expected = {
        "results/official_v6/results.json": "236081e0c721602c742b573e3dfd06a556f9448ddaab299f10007788de51a7c0",
        "results/official_v6/REPORT.md": "bdcdf03ef48aa4cf0a078552938eee853acf9fcde37d6c6f1ae03c0462652189",
        "results/kary_scaling/KARY_VALIDATION_REPORT.json": "5227be3be45ae737efe2c02291c43f6590f8c8636d20007cbfcd22505a7e009c",
        "figures/fig2_official_per_corpus.png": "e21c7c86fa86f9c4240f06f105cfe6fec5c5b2877758935a825a0498b5dea0fc",
        "figures/fig2_official_per_corpus.svg": "7984f57b567d6aa967ed283d7c8319bbc71080eff46ae19d0ab7b8b499852269",
        "figures/fig3_kary_scaling.png": "36699080d40cdf4ccf233a9decc37874843304793e948324b274096d1b9380f7",
        "figures/fig3_kary_scaling.svg": "867e3158deddf972184c200fff3a583b5e2b861d9182efdea2402d0c4979f418",
    }
    assert {name: sha256(ROOT / name) for name in expected} == expected


def test_release_manifest_has_every_required_binary_and_cache_status():
    manifest = json.loads(
        (ROOT / "reproducibility/ARTIFACT_MANIFEST.json").read_text()
    )
    artifacts = manifest["artifacts"]
    assert manifest["release"]["tag"] == "v1-paper"
    assert len(artifacts) == 12
    assert {entry["seed"] for entry in artifacts} == {424242, 271828, 314159}
    roles = [entry["role"] for entry in artifacts]
    assert roles.count("final_clean_v6_checkpoint") == 3
    assert roles.count("continuation_initialization_source_checkpoint") == 3
    assert roles.count("candidate_parent_only_fitted_probe") == 3
    assert roles.count("source_parent_only_fitted_probe") == 3
    assert all(entry["exact_frozen_artifact_hash_match"] is True for entry in artifacts)
    assert manifest["status"]["missing_required_binary_artifacts"] == []
    assert manifest["status"]["caches_included"] is False
    assert {
        entry["byte_size"]
        for entry in manifest["unpublished_regenerable_required_caches"]
    } == {77225389, 363278219}
    expected_checksums = {
        f"{entry['sha256']}  {entry['asset_filename']}" for entry in artifacts
    }
    actual_checksums = set(
        (ROOT / "reproducibility/ARTIFACTS.sha256").read_text().splitlines()
    )
    assert actual_checksums == expected_checksums


def test_official_result_aggregates_are_internally_consistent():
    result = json.loads((ROOT / "results/official_v6/results.json").read_text())
    seeds = result["seeds"]
    seed_auc = [
        seed["safe_arm"]["scopes"]["pooled"]["attacks"]["strongest_orientation_safe_auc"]
        for seed in seeds
    ]
    aggregate = result["primary_three_seed_aggregate"]
    assert len(seeds) == 3
    assert abs(sum(seed_auc) / len(seed_auc) - aggregate["mean_auc"]) < 1e-15
    assert aggregate["ci95"][0] <= aggregate["mean_auc"] <= aggregate["ci95"][1]
    for seed in seeds:
        full = seed["full_document_arm"]
        assert full["every_available_token_processed"] is True
        assert full["truncated_documents"] == 0
        assert full["throughput"]["internal_merges"] == full[
            "builtin_decoder_active_slot_child_cosine"
        ]["observed_internal_merges"]


def test_exact_final_model_constructs_and_runs_a_synthetic_forward():
    cfg = yaml.safe_load((ROOT / "configs/final_v6/seed424242.yaml").read_text())
    assert cfg["encoder_hidden"] == 1024
    assert cfg["encoder_layers"] == 8
    assert cfg["heads"] == 16
    assert cfg["arity"] == 4
    assert cfg["decoder_mode"] == "strict"
    assert cfg["geodesic_parameterization"] == "tangent_expmap"
    assert cfg["max_geodesic_angle"] == 0.5
    model = GeodesicMerge(
        d=768,
        hidden=cfg["encoder_hidden"],
        layers=cfg["encoder_layers"],
        heads=cfg["heads"],
        max_angle=cfg["max_geodesic_angle"],
        decoder=True,
        decoder_layers=cfg["decoder_layers"],
        decoder_length_mode=cfg["decoder_length_mode"],
        importance_mode=cfg["importance_mode"],
        importance_beta=cfg["importance_beta"],
        decoder_mode=cfg["decoder_mode"],
        geodesic_parameterization=cfg["geodesic_parameterization"],
        arity=cfg["arity"],
    ).eval()
    assert len(model.context.layers) == 8
    assert model.context.layers[0].self_attn.num_heads == 16
    assert model.context.layers[0].linear1.out_features == 3072
    children = F.normalize(torch.randn(1, 4, 768), dim=-1)
    lengths = torch.tensor([[1024.0, 1024.0, 1024.0, 17.0]])
    with torch.inference_mode():
        root = model.merge(children, lengths)
        decoded = model.decode(root)
    assert root.shape == (1, 768)
    assert decoded.shape == (1, 4, 768)
    assert torch.isfinite(root).all() and torch.isfinite(decoded).all()
    assert torch.allclose(root.norm(dim=-1), torch.ones(1), atol=2e-5, rtol=0)
    del model
    gc.collect()


def test_k4_active_arities_and_recursive_reduction_smoke():
    torch.manual_seed(17)
    model = GeodesicMerge(
        d=16,
        hidden=32,
        layers=1,
        heads=4,
        dropout=0.0,
        max_angle=0.5,
        decoder=True,
        decoder_layers=1,
        decoder_length_mode="predicted",
        importance_mode="semantic",
        decoder_mode="strict",
        geodesic_parameterization="tangent_expmap",
        arity=4,
    ).eval()
    for active in (2, 3, 4):
        children = F.normalize(torch.randn(1, 4, 16), dim=-1)
        lengths = torch.zeros(1, 4)
        lengths[:, :active] = torch.arange(1, active + 1)
        root = model.merge(children, lengths)
        assert root.shape == (1, 16)
        assert torch.isfinite(root).all()
        assert torch.allclose(root.norm(dim=-1), torch.ones(1), atol=2e-5, rtol=0)

    leaves = F.normalize(torch.randn(1, 9, 16), dim=-1)
    counts = torch.arange(1, 10, dtype=torch.float32).unsqueeze(0)
    root, levels = recursive_reduce(model, leaves, counts)
    assert root.shape == (1, 16)
    assert len(levels) == 2
    assert torch.isfinite(root).all()
    assert torch.allclose(root.norm(dim=-1), torch.ones(1), atol=2e-5, rtol=0)
