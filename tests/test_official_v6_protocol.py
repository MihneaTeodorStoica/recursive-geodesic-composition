from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts/official_v6"
for path in (STAGE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_official_test_v6_recovery as evaluate  # noqa: E402
import official_test_v6_protocol as protocol  # noqa: E402
import prepare_official_test_v6 as prepare  # noqa: E402


def config():
    return yaml.safe_load(protocol.CONFIG.read_text())


def test_config_freezes_range_complete_refusal_partition_and_both_arms():
    cfg = config()
    protocol.validate_config(cfg)
    assert (cfg["scan_start"], cfg["scan_end_exclusive"]) == protocol.SCAN
    assert list(cfg["prior_ranges"]) == list(protocol.PRIOR_RANGES)
    assert list(cfg["prior_ranges"].values())[0][0] == 0
    assert list(cfg["prior_ranges"].values())[-1][1] == protocol.SCAN[0]
    assert cfg["prior_ranges"]["official_v5"] == [110000, 140000]
    assert cfg["prior_ranges"]["clean_v6_fresh_train"] == [140000, 152000]
    assert cfg["prior_ranges"]["clean_v6_fresh_development"] == [152000, 161000]
    assert cfg["prior_ranges"]["mandatory_preaccess_buffer"] == [161000, 162000]
    assert cfg["safe_arm"]["prefix_min_tokens"] == 1025
    assert cfg["safe_arm"]["teacher_max_tokens"] == 4096
    assert cfg["full_arm"] == {
        "every_available_token": True,
        "token_cap": None,
        "maximum_leaves": None,
        "recursion_depth_cap": None,
        "truncation": False,
        "full_root_teacher_target": False,
        "auc": False,
        "teacher_cosine": False,
        "leaf_storage_dtype": "float16",
        "shard_per_source_document": True,
    }
    changed = copy.deepcopy(cfg)
    changed["full_arm"]["maximum_leaves"] = 1
    with pytest.raises(ValueError, match="acquired a cap"):
        protocol.validate_config(changed)


def test_stream_refuses_prior_v5_train_development_buffer_and_partial_windows():
    refused = (
        *protocol.PRIOR_RANGES.values(),
        (protocol.SCAN[0] + 1, protocol.SCAN[1]),
    )
    for interval in refused:
        stream = prepare._stream_scan_window({}, *interval)
        with pytest.raises(PermissionError, match="only .* authorized"):
            next(stream)
    assert "datasets" not in prepare.__dict__


def test_selection_and_exact_chunker_never_drop_a_tail():
    assert protocol.safe_prefix_length(1024) is None
    assert protocol.safe_prefix_length(1025) == 1025
    assert protocol.safe_prefix_length(4096) == 4096
    assert protocol.safe_prefix_length(1000000) == 4096
    for count in (1, 2, 3, 1024, 1025, 1027, 2049, 4096, 4097, 10001):
        chunks = protocol.exact_chunks(list(range(count)))
        assert sum(map(len, chunks)) == count
        assert [value for chunk in chunks for value in chunk] == list(range(count))
        assert all(0 < len(chunk) <= protocol.CHUNK_TOKENS for chunk in chunks)


def synthetic_safe_row(count=4, total=4096):
    lengths = [1024] * (count - 1) + [total - 1024 * (count - 1)]
    source = protocol.SCAN[0]
    return {
        "id": f"fineweb:{source}",
        "source_index": source,
        "selected_rank": 0,
        "source_document_id": "synthetic",
        "corpus": "fineweb",
        "cache_namespace": protocol.CACHE_NAMESPACE,
        "cache_role": protocol.ROLE,
        "arm": "safe_teacher_comparable",
        "arity": protocol.ARITY,
        "reduction": "direct_single_k4",
        "chunk_tokens": protocol.CHUNK_TOKENS,
        "teacher_max_tokens": protocol.SAFE_RANGE[1],
        "safe_prefix_token_count": total,
        "full_token_count": total,
        "safe_prefix_token_ids_sha256": "a",
        "full_token_ids_sha256": "b",
        "leaves": torch.ones(count, 768),
        "lengths": torch.tensor(lengths, dtype=torch.float32),
        "teacher_target": torch.ones(768),
    }


def synthetic_full_shard(leaves=5, tokens=4097):
    lengths = [1024] * (leaves - 1) + [tokens - 1024 * (leaves - 1)]
    source = protocol.SCAN[0]
    metadata = {
        "id": f"fineweb:{source}",
        "source_index": source,
        "source_document_id": "synthetic",
        "corpus": "fineweb",
        "full_token_count": tokens,
        "full_leaf_count": leaves,
        "tree_depth": protocol.reduction_depth(leaves),
    }
    shard = {
        **metadata,
        "cache_namespace": protocol.CACHE_NAMESPACE,
        "cache_role": protocol.ROLE,
        "arm": "unrestricted_full_document_recursion",
        "arity": protocol.ARITY,
        "chunk_tokens": protocol.CHUNK_TOKENS,
        "reduction": "deterministic_contiguous_k4_carry_singleton",
        "truncated": False,
        "token_cap": None,
        "maximum_leaves": None,
        "recursion_depth_cap": None,
        "leaf_storage_dtype": "float16",
        "full_token_ids_sha256": "hash",
        "lengths": torch.tensor(lengths, dtype=torch.int32),
        "leaves": torch.randn(leaves, 768, dtype=torch.float16),
    }
    return shard, metadata


def test_direct_role_finite_exact_count_and_forbidden_comparison_validators():
    safe = synthetic_safe_row()
    prepare.validate_safe_row(safe, "fineweb", protocol.SCAN[0] + 1)
    changed = copy.deepcopy(safe)
    changed["cache_role"] = "validation"
    with pytest.raises(ValueError, match="role"):
        prepare.validate_safe_row(changed, "fineweb", protocol.SCAN[0] + 1)
    changed = copy.deepcopy(safe)
    changed["teacher_target"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        prepare.validate_safe_row(changed, "fineweb", protocol.SCAN[0] + 1)
    shard, metadata = synthetic_full_shard()
    prepare.validate_full_shard(shard, metadata)
    changed = copy.deepcopy(shard)
    changed["teacher_target"] = torch.ones(768)
    with pytest.raises(ValueError, match="forbidden teacher"):
        prepare.validate_full_shard(changed, metadata)
    changed = copy.deepcopy(shard)
    changed["lengths"][-1] -= 1
    with pytest.raises(ValueError, match="exact token"):
        prepare.validate_full_shard(changed, metadata)


class DummyStrictK4(torch.nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.d = dimension
        self.arity = protocol.ARITY
        self.importance_beta = 1.0
        self._last_merge_aux = None

    def merge(self, children, masses):
        active = masses > 0
        weights = masses / masses.sum(1, keepdim=True)
        roots = F.normalize(
            (F.normalize(children, dim=-1) * weights[..., None]).sum(1), dim=-1
        )
        self._last_merge_aux = {
            "active_mask": active,
            "angle": torch.zeros(len(children), device=children.device),
            "geodesic_saturated": torch.zeros(
                len(children), dtype=torch.bool, device=children.device
            ),
            "corrections": torch.zeros_like(masses),
            "effective_weights": weights,
        }
        return roots

    def decode(self, parent):
        return F.normalize(parent[:, None, :].expand(-1, protocol.ARITY, -1), dim=-1)


def telemetry_book():
    return {
        "merge_count": 0,
        "decoded_merge_count": 0,
        "decoded_child_count": 0,
        "geodesic": defaultdict(evaluate.RunningStats),
        "importance": defaultdict(evaluate.RunningStats),
        "effective_weight_max": defaultdict(evaluate.RunningStats),
        "decoder_cosine": defaultdict(evaluate.RunningStats),
        "geodesic_saturation": defaultdict(int),
        "importance_saturation": defaultdict(int),
        "importance_active": defaultdict(int),
    }


@pytest.mark.parametrize(
    "leaf_count,tokens", [(2, 1025), (4, 4096), (5, 4097), (17, 16385), (65, 65537)]
)
def test_unrestricted_recursion_decodes_every_internal_merge(leaf_count, tokens):
    shard, metadata = synthetic_full_shard(leaf_count, tokens)
    prepare.validate_full_shard(shard, metadata)
    book = telemetry_book()
    root = evaluate.reduce_full_document(
        DummyStrictK4(), shard, "fineweb", torch.device("cpu"), 3, book
    )
    expected = evaluate.expected_internal_merges(leaf_count)
    assert torch.isfinite(root).all()
    assert book["merge_count"] == expected == book["decoded_merge_count"]
    assert book["decoded_child_count"] >= leaf_count
    assert book["decoder_cosine"]["aggregate"].count == book["decoded_child_count"]
    assert any(
        str(key).startswith("corpus:fineweb|depth:") for key in book["decoder_cosine"]
    )
    assert any(
        "child_type:recursive_child" in str(key) for key in book["decoder_cosine"]
    ) == (leaf_count > protocol.ARITY)


def synthetic_attack_rows(n=36, dimension=6):
    rng = np.random.default_rng(5)
    teacher = rng.normal(size=(n, dimension))
    generated = teacher + rng.normal(0, 0.05, size=(n, dimension))
    groups = np.asarray([f"doc:{index}" for index in range(n)])
    attack = {
        "split_seed": 1729,
        "model_seed": 1729,
        "test_size": 0.5,
        "logistic_c": 1.0,
        "logistic_max_iter": 1000,
        "rbf_c": 1.0,
        "rbf_gamma": 1.0 / dimension,
    }
    return teacher, generated, groups, attack


def test_fixed_grouped_attacks_and_exact_bootstrap_contract():
    teacher, generated, groups, attack = synthetic_attack_rows()
    result, _ = evaluate.evaluate_fixed_attacks(
        teacher, generated, groups, attack, 2000, 91
    )
    assert tuple(result["attacks"]) == evaluate.ATTACK_NAMES
    assert result["group_overlap"] == 0
    assert all(
        len(value["orientation_safe_auc_document_group_ci95"]) == 2
        for value in result["attacks"].values()
    )
    duplicate = groups.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="unique"):
        evaluate.grouped_attack_split(teacher, generated, duplicate, attack)
    generated[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        evaluate.grouped_attack_split(teacher, generated, groups, attack)
    with pytest.raises(ValueError, match="exactly 2000"):
        evaluate.document_group_bootstrap(
            np.tile([0, 1], 4), np.arange(8), np.repeat(np.arange(4), 2), 1999, 1
        )


def test_three_seed_ci_requires_identical_shared_groups():
    groups = np.repeat(np.asarray([f"g{i}" for i in range(12)]), 2)
    y = np.tile(np.asarray([1, 0]), 12)
    bundle = {"groups": groups, "y": y, "scores": np.tile([0.9, 0.1], 12)}
    point, ci = evaluate.three_seed_shared_ci(
        [bundle] * len(protocol.SEED_ORDER), 2000, 7
    )
    assert point == 1.0 and ci == [1.0, 1.0]
    changed = {**bundle, "groups": groups.copy()}
    changed["groups"][0] = "other"
    with pytest.raises(AssertionError, match="not shared"):
        evaluate.three_seed_shared_ci([bundle, bundle, changed], 2000, 7)


def complete_full(healthy=True, saturation=0):
    health = {
        scope: {
            "all_finite": healthy,
            "collapse": not healthy,
            "output_norm_saturation_count": saturation,
        }
        for scope in evaluate.SCOPES
    }
    aggregations = {"aggregate": {"mean": 0.91}}
    for corpus in protocol.CORPUS_ORDER:
        aggregations[f"corpus:{corpus}"] = {"mean": 0.91}
        aggregations[f"corpus:{corpus}|child_type:leaf_child"] = {"mean": 0.91}
        aggregations[f"corpus:{corpus}|child_type:recursive_child"] = {"mean": 0.91}
    aggregations["depth:1"] = {"mean": 0.91}
    return {
        "every_available_token_processed": True,
        "truncated_documents": 0,
        "token_cap": None,
        "maximum_leaves": None,
        "recursion_depth_cap": None,
        "full_root_teacher_target": False,
        "teacher_comparison_metrics": [],
        "attacks": [],
        "throughput": {
            "documents": len(protocol.CORPUS_ORDER) * protocol.TARGET_DOCUMENTS
        },
        "health": health,
        "all_level_geometry_and_saturation": {
            "aggregate": {
                "geodesic_saturation_count": saturation,
                "importance_saturation_count": saturation,
            }
        },
        "document_distributions": {
            "by_corpus": {corpus: {} for corpus in protocol.CORPUS_ORDER}
        },
        "builtin_decoder_active_slot_child_cosine": {
            "expected_internal_merges": 9,
            "observed_internal_merges": 9,
            "aggregations": aggregations,
        },
    }


def gate_seed(cfg, *, auc=None, cosine=None, child=None, floor=True, healthy=True):
    gate = cfg["reporting_gate"]
    if auc is None:
        auc = float(gate["sole_auc_target"]) - 0.001
    if cosine is None:
        cosine = float(gate["every_seed_pooled_teacher_cosine_strictly_above"]) + 0.001
    if child is None:
        child = (
            float(
                gate["every_seed_every_safe_scope_builtin_exact4_child_cosine_at_least"]
            )
            + 0.001
        )
    safe_scopes = {
        scope: {
            "teacher_cosine": cosine,
            "attacks": {"strongest_orientation_safe_auc": auc},
            "health": {
                "all_finite": healthy,
                "collapse": not healthy,
                "output_norm_saturation_count": 0,
            },
        }
        for scope in evaluate.SCOPES
    }
    return {
        "safe_arm": {
            "scopes": safe_scopes,
            "telemetry": {
                "geodesic_saturation_count": 0,
                "importance_saturation_count": 0,
            },
            "builtin_decoder_exact4": {
                "scopes": {
                    scope: {"ordered_child_cosine": child} for scope in evaluate.SCOPES
                }
            },
            "frozen_v6_train_probe_retention": {
                "scopes": {scope: {"floor_pass": floor} for scope in evaluate.SCOPES}
            },
        },
        "full_document_arm": complete_full(healthy=healthy),
    }


def test_reporting_gate_uses_sole_target_derived_seed_guard_and_never_suppresses():
    cfg = config()
    target = float(cfg["reporting_gate"]["sole_auc_target"])
    seeds = [gate_seed(cfg) for _ in protocol.SEED_ORDER]
    gate = evaluate.reporting_gate(seeds, target - 0.001, cfg)
    assert gate["passed"] is True
    assert gate["suppresses_or_selects_outcomes"] is False
    assert evaluate.reporting_gate(seeds, target, cfg)["passed"] is False
    derived_guard = target + float(cfg["reporting_gate"]["per_seed_guard_margin"])
    guarded = [gate_seed(cfg, auc=derived_guard), gate_seed(cfg), gate_seed(cfg)]
    assert evaluate.reporting_gate(guarded, target - 0.001, cfg)["passed"] is False
    low_cosine = float(
        cfg["reporting_gate"]["every_seed_pooled_teacher_cosine_strictly_above"]
    )
    assert (
        evaluate.reporting_gate(
            [gate_seed(cfg, cosine=low_cosine)] * len(protocol.SEED_ORDER),
            target - 0.001,
            cfg,
        )["passed"]
        is False
    )
    assert (
        evaluate.reporting_gate(
            [gate_seed(cfg, floor=False)] * len(protocol.SEED_ORDER),
            target - 0.001,
            cfg,
        )["passed"]
        is False
    )
    incomplete = gate_seed(cfg)
    incomplete["full_document_arm"]["maximum_leaves"] = 4
    assert (
        evaluate.reporting_gate(
            [incomplete, gate_seed(cfg), gate_seed(cfg)], target - 0.001, cfg
        )["passed"]
        is False
    )


def test_one_shot_claim_and_storage_guard_fail_closed(tmp_path, monkeypatch):
    marker = tmp_path / "STARTED"
    protocol.claim_once(marker)
    with pytest.raises(FileExistsError, match="claimed"):
        protocol.claim_once(marker)

    class Usage:
        free = 100

    monkeypatch.setattr(protocol.shutil, "disk_usage", lambda _: Usage())
    with pytest.raises(RuntimeError, match="refusing to truncate"):
        protocol.require_storage(tmp_path, 101, "synthetic full document")


def test_new_protocol_has_no_competing_numeric_auc_target():
    files = [
        path
        for path in STAGE.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".md", ".sh"}
    ]
    text = "\n".join(path.read_text() for path in files)
    forbidden = ("0." + "58", "." + "579", "0." + "61")
    assert all(value not in text for value in forbidden)
    cfg = config()
    assert set(key for key in cfg["reporting_gate"] if "auc" in key.lower()) == {
        "sole_auc_target"
    }
