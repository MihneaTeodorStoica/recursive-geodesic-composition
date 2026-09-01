"""Fail-closed protocol for the one-shot strict-K4 AE official evaluation v6.

This module cannot import a corpus loader or open an evaluation cache. It verifies
all development, checkpoint, source, probe, range, and storage contracts before a
data entry point may proceed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = ROOT / "configs/official_v6.yaml"
STAGE = "kary4_strict_autoencoder_official_test_v6"
CACHE_NAMESPACE = STAGE
ROLE = "test"
SCAN = (162000, 192000)
SAFE_RANGE = (1025, 4096)
TARGET_DOCUMENTS = 1000
ARITY = 4
CHUNK_TOKENS = 1024
SEED_ORDER = (424242, 271828, 314159)
CORPUS_ORDER = (
    "fineweb_edu",
    "fineweb",
    "openwebtext",
    "wikipedia_en_20231101",
)
PARENT_COMMIT = "f38f75d2d70a9af92d3bdeae959281173c7998ee"
PRIOR_RANGES = {
    "all_earlier_work_and_unallocated_gaps": (0, 110000),
    "official_v5": (110000, 140000),
    "clean_v6_fresh_train": (140000, 152000),
    "clean_v6_fresh_development": (152000, 161000),
    "mandatory_preaccess_buffer": (161000, 162000),
}
# The freeze JSON hashes below identify normalized publication copies. Only an
# irrelevant absolute workspace prefix was removed; scientific values and
# all model/probe/data hashes are unchanged (see reproducibility/provenance.md).
EXPECTED_SEEDS = {
    424242: {
        "checkpoint": "results_kary4_strict_autoencoder_v6_clean_development_seed424242/autoencoder_seed424242.pt",
        "checkpoint_sha256": "989f9bc496bea484974679cd181e43836caf3283ffdc8e66507046945e0902de",
        "source_checkpoint": "results_kary4_strict_autoencoder_seed424_dev_v2/autoencoder_seed424242.pt",
        "source_checkpoint_sha256": "2365a789f61ba6dab9967041b8dcee2a19dc969fd97b71026bdee970f7ea4797",
        "freeze": "results/clean_v6_development/seeds/424242/PRE_DEVELOPMENT_FROZEN.json",
        "freeze_sha256": "33610c1784945864cc102d83f58a4095679cc72e105417be8f8d9d728437549b",
        "candidate_probe": "results_kary4_strict_autoencoder_v6_clean_development_seed424242/train_fitted_matched_probes/candidate_parent_only.pt",
        "candidate_probe_sha256": "a9e30ac9df31af63c525bdf08e812d5562bf2ba7c4c60ea8f439f4e4492e1943",
        "source_probe": "results_kary4_strict_autoencoder_v6_clean_development_seed424242/train_fitted_matched_probes/source_parent_only.pt",
        "source_probe_sha256": "7d71e0951ebace7c65dfbae7ca54a23f9c972d63ba308eb7d4b9f727d9a6e830",
        "frozen_train_probe_floors": {
            "pooled": 0.897565411567688,
            "fineweb_edu": 0.8956167569160461,
            "fineweb": 0.8919562568664551,
            "openwebtext": 0.8993962278366089,
            "wikipedia_en_20231101": 0.9032922854423523,
        },
    },
    271828: {
        "checkpoint": "results_kary4_strict_autoencoder_v6_clean_development_seed271828/autoencoder_seed271828.pt",
        "checkpoint_sha256": "a4cb1d79a7ae1bb0ca8855e3605c1ff50e761ca17ada39aa6b0237900dce7528",
        "source_checkpoint": "results_kary4_strict_autoencoder_multiseed_confirmation_v1_seed271828/autoencoder_seed271828.pt",
        "source_checkpoint_sha256": "63103ce0b940dab6eec02c62ae5e177b0f23941f1775624b2fe3b27f60105603",
        "freeze": "results/clean_v6_development/seeds/271828/PRE_DEVELOPMENT_FROZEN.json",
        "freeze_sha256": "4666b6c263c54f71c5be0a78c37352ec6900d53eb5abfb535c80ad52d8be562c",
        "candidate_probe": "results_kary4_strict_autoencoder_v6_clean_development_seed271828/train_fitted_matched_probes/candidate_parent_only.pt",
        "candidate_probe_sha256": "7eff7a553a7dc17d2f75127b6622067f2f11add3454884b0c532782428e98fb9",
        "source_probe": "results_kary4_strict_autoencoder_v6_clean_development_seed271828/train_fitted_matched_probes/source_parent_only.pt",
        "source_probe_sha256": "b3f422a0cc7c8ff5cb9ce1e066f38b51f004417d6d0d74f4e974f78eeeadbc8e",
        "frozen_train_probe_floors": {
            "pooled": 0.8975575437545776,
            "fineweb_edu": 0.8955862393379211,
            "fineweb": 0.8920076360702515,
            "openwebtext": 0.8994596471786499,
            "wikipedia_en_20231101": 0.9031767120361328,
        },
    },
    314159: {
        "checkpoint": "results_kary4_strict_autoencoder_v6_clean_development_seed314159/autoencoder_seed314159.pt",
        "checkpoint_sha256": "45a80bc3005bdf79fcafe3fce65b927a8f39e7ec2c019a34c3e9d1c8ce327a07",
        "source_checkpoint": "results_kary4_strict_autoencoder_multiseed_confirmation_v1_seed314159/autoencoder_seed314159.pt",
        "source_checkpoint_sha256": "27116bb5dbe279750264147100b38323ec432329b385404f59db5bf753d4b3a2",
        "freeze": "results/clean_v6_development/seeds/314159/PRE_DEVELOPMENT_FROZEN.json",
        "freeze_sha256": "4a510c25ad0287f1e38d9e0ab395beddc760c97b8254b08d466b95a79c07895c",
        "candidate_probe": "results_kary4_strict_autoencoder_v6_clean_development_seed314159/train_fitted_matched_probes/candidate_parent_only.pt",
        "candidate_probe_sha256": "14736e8cbdc623cd004ef031f8af719dffa0e627c37425611e9ede6df2bd514a",
        "source_probe": "results_kary4_strict_autoencoder_v6_clean_development_seed314159/train_fitted_matched_probes/source_parent_only.pt",
        "source_probe_sha256": "617cd5430ed9751121e3f222de942ea8f4e5f4164ce9a0514e36ae88bb954497",
        "frozen_train_probe_floors": {
            "pooled": 0.8975992670059204,
            "fineweb_edu": 0.8955495228767395,
            "fineweb": 0.8919384350776672,
            "openwebtext": 0.8995260467529297,
            "wikipedia_en_20231101": 0.9033831825256348,
        },
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def require_hash(path: str | Path, expected: str, label: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def half_open_overlap(left, right) -> bool:
    a, b = map(int, left)
    c, d = map(int, right)
    if min(a, c) < 0 or b <= a or d <= c:
        raise ValueError("ranges must be nonempty nonnegative half-open intervals")
    return max(a, c) < min(b, d)


def validate_ranges(prior: dict, scan) -> None:
    declared = {str(key): tuple(map(int, value)) for key, value in prior.items()}
    if declared != PRIOR_RANGES:
        raise ValueError("refused source partition changed")
    if tuple(map(int, scan)) != SCAN:
        raise PermissionError(f"v6 authorizes only untouched source scan {SCAN}")
    cursor = 0
    for name, interval in declared.items():
        if interval[0] != cursor:
            raise ValueError(f"refused ranges are not contiguous before {name}")
        cursor = interval[1]
        if half_open_overlap(interval, SCAN):
            raise ValueError(f"v6 scan overlaps refused {name}: {interval}")
    if cursor != SCAN[0]:
        raise ValueError("refused ranges do not cover every source index before scan")


def exact_chunks(token_ids: list[int], width: int = CHUNK_TOKENS) -> list[list[int]]:
    """Return exact contiguous coverage, retaining every final short leaf."""
    if int(width) != CHUNK_TOKENS:
        raise ValueError("v6 chunk width is frozen")
    chunks = [
        token_ids[start : start + width] for start in range(0, len(token_ids), width)
    ]
    if any(not chunk or len(chunk) > width for chunk in chunks):
        raise AssertionError("exact chunker emitted an empty or oversized leaf")
    if sum(map(len, chunks)) != len(token_ids):
        raise AssertionError("exact chunker did not cover every token")
    return chunks


def safe_prefix_length(total_tokens: int) -> int | None:
    count = int(total_tokens)
    if count < SAFE_RANGE[0]:
        return None
    return min(count, SAFE_RANGE[1])


def reduction_depth(leaves: int, arity: int = ARITY) -> int:
    count, depth = int(leaves), 0
    if count < 1 or int(arity) != ARITY:
        raise ValueError("depth requires at least one leaf and exact K4")
    while count > 1:
        count = (count + arity - 1) // arity
        depth += 1
    return depth


def validate_config(cfg: dict) -> None:
    if cfg.get("stage") != STAGE or int(cfg.get("protocol_version", -1)) != 6:
        raise ValueError("wrong v6 stage/protocol")
    if cfg.get("implementation_parent_commit") != PARENT_COMMIT:
        raise ValueError("v6 must remain based on the committed clean development")
    if (
        cfg.get("frozen_before_official_access") is not True
        or cfg.get("official_test_role") != ROLE
    ):
        raise ValueError("v6 must be frozen and test-role-only")
    if (
        cfg.get("cache_dir") != f"cache_{STAGE}"
        or cfg.get("output_dir") != f"results_{STAGE}"
    ):
        raise ValueError("v6 direct cache/result namespace changed")
    if (
        int(cfg.get("arity", -1)),
        int(cfg.get("chunk_tokens", -1)),
        cfg.get("reduction"),
    ) != (ARITY, CHUNK_TOKENS, "deterministic_contiguous_k4_carry_singleton"):
        raise ValueError("v6 requires strict contiguous K4")
    if (
        cfg.get("selection") != "first_1000_safe_prefix_length_eligible_only"
        or int(cfg.get("eligible_documents_per_corpus", -1)) != TARGET_DOCUMENTS
    ):
        raise ValueError("v6 deterministic selection changed")
    validate_ranges(
        cfg.get("prior_ranges", {}),
        (cfg.get("scan_start"), cfg.get("scan_end_exclusive")),
    )
    if cfg.get("safe_arm") != {
        "prefix_min_tokens": 1025,
        "prefix_max_tokens": 4096,
        "teacher_max_tokens": 4096,
        "direct_root_only": True,
        "no_truncation_beyond_declared_prefix": True,
    }:
        raise ValueError("safe direct teacher-comparable arm changed")
    if cfg.get("full_arm") != {
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
    }:
        raise ValueError("unrestricted full-document arm acquired a cap/comparison")
    corpora = cfg.get("corpora", ())
    if tuple(str(item.get("id")) for item in corpora) != CORPUS_ORDER:
        raise ValueError("corpus order changed")
    if any(float(item.get("weight", 0)) != 0.25 for item in corpora):
        raise ValueError("corpus weights must remain equal")
    if len(
        {
            (
                item.get("dataset_name"),
                item.get("dataset_config"),
                item.get("dataset_revision"),
            )
            for item in corpora
        }
    ) != len(CORPUS_ORDER):
        raise ValueError("duplicate or incomplete corpus pins")
    configured = {int(item.get("seed", -1)): item for item in cfg.get("seeds", ())}
    if tuple(configured) != SEED_ORDER:
        raise ValueError("seed order changed")
    for seed, expected in EXPECTED_SEEDS.items():
        if configured[seed] != {"seed": seed, **expected}:
            raise ValueError(f"seed {seed} local artifacts or floors changed")
    development = cfg.get("development_provenance", {})
    exact_development = {
        "final_results": "results/clean_v6_development/FINAL_RESULTS.json",
        "final_results_sha256": "a879106602e3b9f31517e29ca6cfccf80f86bbfb20589d263bbaf0a66e9a04bb",
        "post_final_verification": "results/clean_v6_development/POST_FINAL_VERIFICATION.json",
        "post_final_verification_sha256": "a095a61f1350b8ba347a1b88fd5260eb964cfa36ce4924623f2809e4de391b19",
        "checkpoint_list": "results/clean_v6_development/CHECKPOINTS.sha256",
        "checkpoint_list_sha256": "7f82643c44146d8d3cb151fa547ba2f09aef63dc15f35210589f145dd73ca1f5",
        "local_artifact_list": "results/clean_v6_development/LOCAL_LARGE_ARTIFACTS.sha256",
        "local_artifact_list_sha256": "6894a9304e74f991bdb28031a1ca813744e80e7403ea3db33aa1d547e3d59ab5",
        "require_development_pass": True,
        "require_no_protected_access": True,
        "required_training_updates": 400,
        "required_decoder_only_updates": 300,
        "required_joint_updates": 100,
    }
    if development != exact_development:
        raise ValueError("clean-development provenance declaration changed")
    if cfg.get("attack_config") != "configs/evaluator_attacks_v1.yaml":
        raise ValueError("fixed attack path changed")
    if cfg.get("attack_config_sha256") != (
        "49e7a2996e33482239819430e92426fc1e4f9ff88628dfd2ac74f6da94726800"
    ):
        raise ValueError("fixed attack hash changed")
    bootstrap = cfg.get("bootstrap", {})
    if int(bootstrap.get("reps", -1)) != 2000:
        raise ValueError("every bootstrap must remain at 2000 replicates")
    if tuple(bootstrap.get("per_seed_seeds", {})) != SEED_ORDER:
        raise ValueError("per-seed bootstrap seeds changed")
    if tuple(bootstrap.get("three_seed_shared_seeds", {})) != (
        "pooled",
        *CORPUS_ORDER,
    ):
        raise ValueError("shared bootstrap scopes changed")
    if cfg.get("train_probe") != {
        "protocol": "frozen_candidate_specific_fresh_v6_train_parent_only_probe_exact_k4",
        "exact_arity": 4,
        "hidden": 512,
        "fit_role": "fresh_v6_train",
        "updates_on_official": 0,
        "batch_size": 512,
    }:
        raise ValueError("frozen TRAIN probe protocol changed")
    gate = cfg.get("reporting_gate", {})
    if gate != {
        "reporting_only": True,
        "suppress_results": False,
        "sole_auc_target": 0.60,
        "per_seed_guard_margin": 0.01,
        "every_seed_pooled_teacher_cosine_strictly_above": 0.98,
        "every_seed_every_safe_scope_builtin_exact4_child_cosine_at_least": 0.90,
        "require_every_seed_every_frozen_train_probe_floor_pass": True,
        "require_no_safe_or_full_collapse_nonfinite_or_saturation": True,
        "require_full_arm_complete": True,
    }:
        raise ValueError("reporting-only gate changed")
    if cfg.get("reporting") != {
        "report_all_outcomes": True,
        "post_access_tuning_selection_rerun_or_suppression": False,
    }:
        raise ValueError("mandatory all-outcome reporting changed")
    if cfg.get("storage", {}).get("fail_instead_of_truncate") is not True:
        raise ValueError("storage failures must never become semantic truncation")


def load_config(path: str | Path = CONFIG) -> dict:
    path = Path(path).resolve()
    if path != CONFIG.resolve():
        raise PermissionError(f"v6 accepts only its frozen config path: {CONFIG}")
    cfg = yaml.safe_load(path.read_text())
    validate_config(cfg)
    return cfg


def _require_no_protected_access(value: dict, label: str) -> None:
    forbidden_true = (
        "protected_data_accessed",
        "future_protected_opened",
        "reserve_buffer_opened",
        "protected_work_performed",
    )
    for key in forbidden_true:
        if value.get(key, False) is not False:
            raise ValueError(f"{label} records protected access: {key}")


def _validate_parameter_change(change: dict, label: str) -> None:
    if (
        change.get("nonzero") is not True
        or int(change.get("parameter_tensors", 0)) <= 0
        or int(change.get("changed_parameter_tensors", -1))
        != int(change.get("parameter_tensors", -2))
        or not math.isfinite(float(change.get("l2_change", float("nan"))))
        or float(change.get("l2_change", 0)) <= 0
    ):
        raise ValueError(f"invalid genuine nonzero {label} change provenance")


def _validate_model_checkpoint(path: Path, expected_seed: int, label: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state, model_cfg = checkpoint.get("state_dict"), checkpoint.get("config", {})
    checkpoint_seed = int(checkpoint.get("seed", model_cfg.get("seed", -1)))
    if (
        checkpoint.get("variant") != "autoencoder"
        or model_cfg.get("decoder_mode") != "strict"
        or int(model_cfg.get("arity", -1)) != ARITY
        or checkpoint_seed != int(expected_seed)
        or not isinstance(state, dict)
    ):
        raise ValueError(f"{label} is not the frozen decoder-bearing strict-K4 AE")
    decoder_keys = [key for key in state if key.startswith("strict_decoder.")]
    encoder_keys = [key for key in state if not key.startswith("strict_decoder.")]
    if not decoder_keys or not encoder_keys:
        raise ValueError(f"{label} lacks encoder or strict decoder state")
    if any(
        not torch.isfinite(value).all()
        for value in state.values()
        if torch.is_tensor(value)
    ):
        raise FloatingPointError(f"{label} contains nonfinite parameters")
    return {
        "variant": checkpoint["variant"],
        "decoder_mode": model_cfg["decoder_mode"],
        "selected_step": int(checkpoint.get("selected_step", -1)),
        "encoder_state_tensors": len(encoder_keys),
        "decoder_state_tensors": len(decoder_keys),
    }


def validate_immutable_provenance(
    cfg: dict, *, check_checkpoint_tensors: bool = True
) -> dict:
    """Verify clean development and every actual local model/probe before access."""
    development = cfg["development_provenance"]
    for key, label in (
        ("final_results", "clean development final results"),
        ("post_final_verification", "clean development post-final verification"),
        ("checkpoint_list", "clean development checkpoint list"),
        ("local_artifact_list", "clean development local artifact list"),
    ):
        require_hash(ROOT / development[key], development[f"{key}_sha256"], label)
    final = json.loads((ROOT / development["final_results"]).read_text())
    post = json.loads((ROOT / development["post_final_verification"]).read_text())
    if (
        final.get("development_only") is not True
        or final.get("promotion", {}).get("passed") is not True
        or post.get("promotion_passed") is not True
        or post.get("append_only_post_final_verification") is not True
    ):
        raise ValueError("committed clean-v6 development is not a passing final run")
    _require_no_protected_access(final, "clean development final")
    _require_no_protected_access(post, "clean development post-final verification")
    final_by_seed = {int(item["seed"]): item for item in final.get("seeds", ())}
    if set(final_by_seed) != set(SEED_ORDER):
        raise ValueError("clean development does not contain exactly three seeds")
    proof = {}
    configured = {int(item["seed"]): item for item in cfg["seeds"]}
    for seed in SEED_ORDER:
        entry = configured[seed]
        expected = EXPECTED_SEEDS[seed]
        freeze_path = ROOT / entry["freeze"]
        require_hash(freeze_path, entry["freeze_sha256"], f"seed {seed} freeze")
        freeze = json.loads(freeze_path.read_text())
        if (
            freeze.get("development_only") is not True
            or freeze.get("development_cache_opened") is not False
            or freeze.get("selection") != "fixed_final_step_400_without_validation"
            or freeze.get("continued_autoencoder_retraining_not_from_scratch")
            is not True
        ):
            raise ValueError(f"seed {seed} freeze is not pre-development and fixed")
        _require_no_protected_access(freeze, f"seed {seed} freeze")
        if freeze.get("checkpoint") != {
            "path": entry["checkpoint"],
            "sha256": entry["checkpoint_sha256"],
            "step": 400,
        }:
            raise ValueError(f"seed {seed} frozen checkpoint declaration changed")
        if freeze.get("source_checkpoint") != {
            "path": entry["source_checkpoint"],
            "sha256": entry["source_checkpoint_sha256"],
        }:
            raise ValueError(f"seed {seed} source checkpoint declaration changed")
        final_checkpoint = final_by_seed[seed].get("checkpoint", {})
        if final_checkpoint != freeze["checkpoint"]:
            raise ValueError(f"seed {seed} final/freeze checkpoint mismatch")
        training = freeze.get("training", {})
        required_counts = (
            int(development["required_training_updates"]),
            int(development["required_decoder_only_updates"]),
            int(development["required_joint_updates"]),
        )
        if (
            (
                int(training.get("updates", -1)),
                int(training.get("decoder_only_updates", -1)),
                int(training.get("joint_updates", -1)),
            )
            != required_counts
            or training.get("all_numeric_finite") is not True
            or float(training.get("maximum_geodesic_saturation_fraction", -1)) != 0
            or float(training.get("maximum_importance_saturation_fraction", -1)) != 0
        ):
            raise ValueError(f"seed {seed} training provenance changed")
        _validate_parameter_change(training.get("encoder_change", {}), "encoder")
        _validate_parameter_change(training.get("decoder_change", {}), "decoder")
        matched = freeze.get("matched_probe", {})
        expected_probes = {
            "candidate": {
                "path": entry["candidate_probe"],
                "sha256": entry["candidate_probe_sha256"],
            },
            "source": {
                "path": entry["source_probe"],
                "sha256": entry["source_probe_sha256"],
            },
        }
        if (
            matched.get("fit_role") != "fresh_v6_train"
            or int(matched.get("development_updates", -1)) != 0
            or matched.get("tensor_identical_initialization") is not True
            or matched.get("identical_minibatch_schedule") is not True
            or matched.get("floor_rule")
            != "source_probe_disjoint_train_floor_score_minus_fixed_margin"
            or matched.get("probe_artifacts") != expected_probes
            or matched.get("frozen_floors") != entry["frozen_train_probe_floors"]
        ):
            raise ValueError(f"seed {seed} TRAIN probe/floor provenance changed")
        checkpoint_path = ROOT / entry["checkpoint"]
        source_path = ROOT / entry["source_checkpoint"]
        require_hash(checkpoint_path, entry["checkpoint_sha256"], f"seed {seed} AE")
        require_hash(
            source_path,
            entry["source_checkpoint_sha256"],
            f"seed {seed} source AE",
        )
        model_proof = {}
        if check_checkpoint_tensors:
            model_proof = _validate_model_checkpoint(
                checkpoint_path, seed, f"seed {seed} AE"
            )
            _validate_model_checkpoint(source_path, seed, f"seed {seed} source AE")
        probe_proof = {}
        for arm in ("candidate", "source"):
            path = ROOT / expected_probes[arm]["path"]
            require_hash(
                path, expected_probes[arm]["sha256"], f"seed {seed} {arm} probe"
            )
            probe = torch.load(path, map_location="cpu", weights_only=True)
            if (
                probe.get("arm") != arm
                or probe.get("fit_role") != "fresh_v6_train"
                or int(probe.get("seed", -1)) != seed
                or probe.get("selection_audit") != matched.get("selection_audit")
                or probe.get("schedule_sha256") != matched.get("schedule_sha256")
                or not isinstance(probe.get("state_dict"), dict)
                or any(
                    not torch.isfinite(value).all()
                    for value in probe["state_dict"].values()
                )
            ):
                raise ValueError(f"seed {seed} {arm} probe tensor provenance changed")
            probe_proof[arm] = expected_probes[arm]
        proof[seed] = {
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "source_checkpoint": entry["source_checkpoint"],
            "source_checkpoint_sha256": entry["source_checkpoint_sha256"],
            "freeze": entry["freeze"],
            "freeze_sha256": entry["freeze_sha256"],
            "training": training,
            "model": model_proof,
            "train_probes": probe_proof,
            "frozen_train_probe_floors": expected["frozen_train_probe_floors"],
        }
    require_hash(
        ROOT / cfg["attack_config"], cfg["attack_config_sha256"], "fixed attack config"
    )
    return {
        "implementation_parent_commit": PARENT_COMMIT,
        "development_final_results": development["final_results"],
        "development_final_results_sha256": development["final_results_sha256"],
        "post_final_verification": development["post_final_verification"],
        "post_final_verification_sha256": development["post_final_verification_sha256"],
        "protected_data_accessed_before_freeze": False,
        "checkpoints": proof,
        "attack_config_sha256": cfg["attack_config_sha256"],
    }


def require_storage(path: Path, required_bytes: int, context: str) -> int:
    required = int(required_bytes)
    if required < 0:
        raise ValueError("storage requirement must be nonnegative")
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError(
            f"storage guard failed before {context}: free={free} required={required}; refusing to truncate"
        )
    return free


def claim_once(marker: Path, forbidden_targets=()) -> None:
    if any(Path(target).exists() for target in forbidden_targets):
        raise FileExistsError("one-shot target already exists")
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise FileExistsError(f"one-shot stage already claimed: {marker}") from exc
    with os.fdopen(descriptor, "w") as handle:
        handle.write("strict-K4 AE official-v6 one-shot claim; rerun forbidden\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
