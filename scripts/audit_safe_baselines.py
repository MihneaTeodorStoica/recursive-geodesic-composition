#!/usr/bin/env python3
"""Independently check saved safe-baseline scores, splits, provenance and tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from safe_baselines_protocol import (
    DEFAULT_CONFIG, ROOT, PUBLISHED_HASH, atomic_json, checkpoint_entries, load_protocol,
    load_safe_cache, require_hash, sha256, stable_hash,
)
from recursive_geodesic_composition.safe_baselines import BASELINES, LEARNED, CONTRASTS, fidelity_summary


def audit(config_path):
    cfg, official, published = load_protocol(config_path)
    root = ROOT / cfg["output_dir"]
    result = json.loads((root / "RESULTS.json").read_text())
    manifest = json.loads((root / "INFERENCE_MANIFEST.json").read_text())
    require_hash(root / "INFERENCE_MANIFEST.json", result["inference_manifest_sha256"])
    require_hash(root / "DOCUMENT_SCORES.json", result["document_scores_sha256"])
    require_hash(root / "vectors.pt", manifest["vectors_sha256"])
    for path, digest in result["source_sha256"].items():
        require_hash(ROOT / path, digest)
    rows, _ = load_safe_cache(cfg, published)
    payload = torch.load(root / "vectors.pt", map_location="cpu", weights_only=True)
    ids = [row["id"] for row in rows]
    assert ids == payload["groups"]
    assert len(ids) == len(set(ids)) == 4000
    assert torch.equal(torch.stack([row["teacher_target"] for row in rows]), payload["targets"])
    # Independently calculate cosine with PyTorch double precision, rather than
    # trusting the evaluator's NumPy reduction or the serialized scalar scores.
    raw = {key: torch.nn.functional.cosine_similarity(values.double(), payload["targets"].double(), dim=-1).numpy()
           for key, values in payload["vectors"].items()}
    saved = json.loads((root / "DOCUMENT_SCORES.json").read_text())
    assert [row["id"] for row in saved] == ids
    for method, scores in raw.items():
        np.testing.assert_allclose(scores, [row["cosines"][method] for row in saved], atol=1e-14, rtol=0)
    masks = {"pooled": np.ones(len(ids), dtype=bool)}
    masks.update({corpus: np.asarray(payload["labels"]) == corpus for corpus in published["corpus_order"]})
    masks.update({f"arity_{k}": np.asarray(payload["active_children"]) == k for k in (2, 3, 4)})
    scores_by_seed = {str(seed): {method: raw[f"{method}_seed{seed}"] if method in LEARNED else raw[method]
                                 for method in cfg["methods"]} for seed in cfg["seeds"]}
    scores_by_seed["three_seed_mean"] = {method: np.mean([scores_by_seed[str(seed)][method] for seed in cfg["seeds"]], axis=0)
                                         for method in cfg["methods"]}
    for seed, scores in scores_by_seed.items():
        for offset, (scope, mask) in enumerate(masks.items()):
            expected = fidelity_summary(scores, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["paired_seed"] + offset)
            actual = result["fidelity"][seed][scope]
            for section in ("methods", "contrasts"):
                for key in expected[section]:
                    for field in ("mean", "ci95"):
                        np.testing.assert_allclose(expected[section][key][field], actual[section][key][field], atol=1e-14, rtol=0)
    for key, (a, b) in CONTRASTS.items():
        expected = scores_by_seed["three_seed_mean"][a] - scores_by_seed["three_seed_mean"][b]
        np.testing.assert_allclose(expected, [row["three_seed_mean_contrasts"][key] for row in saved], atol=1e-14, rtol=0)
    selected_aucs = {}
    for path, digest in result["attack_result_sha256"].items():
        require_hash(root / path, digest)
        entry = json.loads((root / path).read_text())
        train = set(entry["classifier_train_groups"])
        held = set(entry["classifier_holdout_groups"])
        assert not train & held
        scope_ids = set(np.asarray(ids)[masks[entry["scope"]]])
        assert train | held == scope_ids
        original = published["seeds"][0]["safe_arm"]["scopes"][entry["scope"]]["attacks"]
        assert stable_hash(sorted(train)) == original["train_group_sha256"]
        assert stable_hash(sorted(held)) == original["holdout_group_sha256"]
        folded = {}
        for name, bundle in entry["bundles"].items():
            groups, y = np.asarray(bundle["groups"]), np.asarray(bundle["y"])
            assert set(groups) == held
            for group in held:
                assert sorted(y[groups == group]) == [0, 1]
            auc = roc_auc_score(y, bundle["scores"])
            folded[name] = max(auc, 1 - auc)
            assert abs(auc - entry["metrics"]["attacks"][name]["raw_auc"]) < 1e-14
            assert abs(folded[name] - entry["metrics"]["attacks"][name]["orientation_safe_auc"]) < 1e-14
        assert max(folded, key=folded.get) == entry["metrics"]["strongest_attack"]
        assert abs(max(folded.values()) - entry["metrics"]["strongest_orientation_safe_auc"]) < 1e-14
        selected_aucs[(entry["representation"], entry["scope"])] = max(folded.values())
    assert len(selected_aucs) == 45
    for method in cfg["methods"]:
        for scope in result["detectability"][method]:
            expected = selected_aucs[(method, scope)] if method in BASELINES else np.mean(
                [selected_aucs[(f"{method}_seed{seed}", scope)] for seed in cfg["seeds"]])
            assert abs(expected - result["detectability"][method][scope]["mean_auc"]) < 1e-14
    entries = checkpoint_entries(cfg)
    assert [entry["sha256"] for entry in entries] == [entry["sha256"] for entry in manifest["checkpoints"]]
    require_hash(ROOT / cfg["published_results"], PUBLISHED_HASH)
    verdict = {
        "passed": True, "documents": 4000, "representations": len(raw), "classifier_scopes": 45,
        "checks": {
            "regenerated_population_matches_published_identity": True,
            "input_targets_match_cache": True, "cosines_independently_recomputed_in_torch_float64": True,
            "paired_means_and_intervals_recomputed": True, "per_document_contrasts_verified": True,
            "classifier_splits_match_original_and_keep_pairs": True,
            "classifier_aucs_recomputed_from_saved_predictions": True,
            "three_checkpoint_auc_aggregation_verified": True,
            "final_checkpoints_and_official_results_unchanged": True,
            "analysis_source_and_artifact_hashes_verified": True,
        },
        "results_sha256": sha256(root / "RESULTS.json"), "audit_implementation_sha256": sha256(__file__),
        "note": "This audit does not refit classifiers or resample their AUC intervals. It verifies AUC points, pairing, family choices and aggregation; fidelity intervals are fully recomputed.",
    }
    atomic_json(root / "AUDIT.json", verdict)
    hashes = json.loads((root / "ARTIFACT_HASHES.json").read_text())
    hashes["AUDIT.json"] = sha256(root / "AUDIT.json")
    atomic_json(root / "ARTIFACT_HASHES.json", hashes)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit(parser.parse_args().config)
