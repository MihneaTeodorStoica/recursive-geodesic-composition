"""Checks of committed analysis artifacts; no local cache/checkpoint/GPU required."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/safe_baselines_v1"


def test_revision_artifact_hashes_and_audit():
    hashes = json.loads((OUTPUT / "ARTIFACT_HASHES.json").read_text())
    for name, expected in hashes.items():
        path = OUTPUT / name
        if name == "vectors.pt" and not path.exists():
            continue  # Regenerable local binary, explicitly not a Git object.
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, name
    audit = json.loads((OUTPUT / "AUDIT.json").read_text())
    assert audit["passed"]
    assert audit["documents"] == 4000
    assert audit["representations"] == 9
    assert audit["classifier_scopes"] == 45


def test_paper_table_is_generated_from_computed_results():
    result = json.loads((OUTPUT / "RESULTS.json").read_text())
    methods = result["fidelity"]["three_seed_mean"]["pooled"]["methods"]
    table = (OUTPUT / "safe_baselines_table.tex").read_text()
    names = {"first_chunk": "First chunk", "unweighted_mean": "Unweighted mean",
             "token_weighted_mean": "Token-weighted mean", "learned_reference": "Learned reference",
             "learned_parent": "Final learned parent"}
    for key, label in names.items():
        cosine = methods[key]["mean"]
        expected = [label, f"{cosine:.4f}",
                    f"{cosine - methods['token_weighted_mean']['mean']:+.6f}",
                    f"{cosine - methods['unweighted_mean']['mean']:+.6f}",
                    f"{result['detectability'][key]['pooled']['mean_auc']:.4f}",
                    f"{result['timing'][key]['mean_checkpoint_median_ms_per_document']:.4f}"]
        assert " & ".join(expected) in table
    paper = (ROOT / "paper/main.tex").read_text()
    assert r"\input{../results/safe_baselines_v1/safe_baselines_table.tex}" in paper
    for contrast in result["fidelity"]["three_seed_mean"]["pooled"]["contrasts"].values():
        assert f"{contrast['mean']:+.6f}" in paper
        assert f"[{contrast['ci95'][0]:.6f},{contrast['ci95'][1]:.6f}]" in paper


def test_document_scores_reproduce_published_aggregate_and_pairing():
    result = json.loads((OUTPUT / "RESULTS.json").read_text())
    rows = json.loads((OUTPUT / "DOCUMENT_SCORES.json").read_text())
    assert len(rows) == len({row["id"] for row in rows}) == 4000
    methods = result["fidelity"]["three_seed_mean"]["pooled"]["methods"]
    for method in result["configuration"]["methods"]:
        if method.startswith("learned_"):
            scores = [np.mean([row["cosines"][f"{method}_seed{seed}"] for seed in result["configuration"]["seeds"]]) for row in rows]
        else:
            scores = [row["cosines"][method] for row in rows]
        assert abs(np.mean(scores) - methods[method]["mean"]) < 1e-14
    for name, entry in result["fidelity"]["three_seed_mean"]["pooled"]["contrasts"].items():
        assert abs(np.mean([row["three_seed_mean_contrasts"][name] for row in rows]) - entry["mean"]) < 1e-14
