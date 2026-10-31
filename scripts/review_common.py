"""Integrity and publication helpers for the remaining additive review controls."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from evaluate_continuation_comparison import load_protocol as previous_protocol
from evaluate_reconstruction_calibration import load_inference as calibration_inputs
from safe_baselines_protocol import ROOT, atomic_json, atomic_tensor, sha256, require_hash, stable_hash, configure_device, environment, load_safe_cache, checkpoint_entries
from recursive_geodesic_composition.continuation_comparison import paired_summary
from recursive_geodesic_composition.safe_baselines import cosine_rows, validate_vectors
from recursive_geodesic_composition.evaluate_kary_validation import build_model

CONFIG = ROOT / "configs/review_completion_v1.yaml"
PREVIOUS = {"results/official_v6/results.json": "2cabef20750aa0a80148784012a1518ea2d1c1008ac7d617467b68f4b9cfb917",
 "results/safe_baselines_v1/RESULTS.json": "f16d23f504d3155dce680243a680d6c26ed2835b46f6aef9100b7fd9c1c57da5",
 "results/recursive_reuse_v1/RESULTS.json": "74944d0960b9311ef8b897ef130474b6a04d463cac4ea6222cb8d96aa6614e29",
 "results/reconstruction_calibration_v1/RESULTS.json": "6f378128f85bcac3ae762c9a931985e3fd5a110d08cc8334bcfef87746bc46c7",
 "results/continuation_comparison_v1/RESULTS.json": "84c151264e962577a35cd12ad47ace1a63852d5abcc8ddcaa1784827d23b16c1"}


def context():
    for path, digest in PREVIOUS.items():
        require_hash(ROOT / path, digest)
    cfg = yaml.safe_load(CONFIG.read_text())
    _, cal, baseline, official, published = previous_protocol()
    return cfg, cal, baseline, official, published


def begin(name, script, extra=(), device_name="cuda", cache=False):
    cfg, cal, baseline, official, published = context()
    device = configure_device(cfg, device_name)
    root = ROOT / (f"cache_{name}_v1" if cache else f"results/{name}_v1")
    if root.exists():
        raise FileExistsError(f"create-only study: {root}")
    files = [CONFIG, Path(__file__), Path(script), ROOT / "src/recursive_geodesic_composition/extended_controls.py",
             ROOT / "src/recursive_geodesic_composition/architecture.py", ROOT / "src/recursive_geodesic_composition/data.py",
             ROOT / "src/recursive_geodesic_composition/evaluate_kary_validation.py", *map(Path, extra)]
    protocol = {"study": name, "configuration": cfg, "config_sha256": sha256(CONFIG), "started_unix": time.time(),
                "source_sha256": {str(p.resolve().relative_to(ROOT)): sha256(p) for p in files},
                "immutable_previous_results": PREVIOUS, "environment": environment(device)}
    root.mkdir(parents=True)
    atomic_json(root / "PROTOCOL.json", protocol)
    return root, cfg, device, cal, baseline, official, published


def verify_protocol(root):
    p = json.loads((root / "PROTOCOL.json").read_text())
    for section in ("source_sha256", "immutable_previous_results"):
        for name, digest in p[section].items():
            require_hash(ROOT / name, digest)
    return p


def seal(root, result):
    protocol = verify_protocol(root)
    if (root / "RESULTS.json").exists():
        raise FileExistsError("results are immutable")
    atomic_json(root / "RESULTS.json", result | {"protocol_sha256": sha256(root / "PROTOCOL.json"), "completed_unix": time.time()})
    hashes(root)


def hashes(root):
    atomic_json(root / "ARTIFACT_HASHES.json", {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "ARTIFACT_HASHES.json"})


def exact_inputs(cal, baseline, published):
    payload, manifest = calibration_inputs(cal)
    rows, _ = load_safe_cache(baseline, published)
    rows = [r for r in rows if len(r["lengths"]) == 4]
    assert [r["id"] for r in rows] == payload["groups"]
    assert torch.equal(torch.stack([r["leaves"] for r in rows]), payload["children"])
    return payload, rows, manifest


def summarize(columns, labels, cfg, seed_offset=0):
    scopes = {"pooled": np.ones(len(labels), bool)} | {name: np.asarray(labels) == name for name in dict.fromkeys(labels)}
    return {scope: paired_summary(columns, mask, cfg["bootstrap"]["reps"], cfg["bootstrap"]["seed"] + seed_offset+i) for i, (scope, mask) in enumerate(scopes.items())}


def save_scores(root, scores, groups, labels, cfg, offset=0):
    # Per-checkpoint (or probe-fit) scores have common document identities.
    scores = {str(s): {k: np.asarray(v) for k, v in cols.items()} for s, cols in scores.items()}
    keys = list(next(iter(scores.values())))
    assert all(set(cols) == set(keys) for cols in scores.values())
    scores["mean"] = {k: np.mean([cols[k] for cols in scores.values()], axis=0) for k in keys}
    atomic_json(root / "SCORES.json", {"groups": groups, "labels": labels,
        "scores": {s: {k: v.tolist() for k, v in cols.items()} for s, cols in scores.items()}})
    return {s: summarize(cols, labels, cfg, offset) for s, cols in scores.items()}


def make_teacher(official, device):
    from recursive_geodesic_composition.data import FrozenTeacher
    teacher = FrozenTeacher(official["model_name"], official["model_revision"], str(device))
    assert {p.dtype for p in teacher.model.parameters()} == {torch.float16}
    assert teacher.model.config.hidden_size == 768
    return teacher


def read_tokens(baseline, published):
    result = []
    for corpus in published["corpus_order"]:
        path = ROOT / baseline["cache_dir"] / corpus / "selected_tokens.pt"
        m = json.loads(path.with_suffix(".json").read_text())
        require_hash(path, m["sha256"])
        result.extend(torch.load(path, weights_only=True, map_location="cpu"))
    return result


def report_table(root, title, rows, caption, label):
    # rows: (label, mean, ci95). Full-precision results are never edited here.
    lines = [f"# {title}", "", "| Measurement | Mean | Paired 95% CI |", "| --- | ---: | --- |"]
    tex = [r"\begin{table}[H]", r"\centering\small", r"\begin{tabularx}{\linewidth}{Yrr}", r"\toprule",
           r"Measurement & Mean & 95\% CI \\", r"\midrule"]
    for name, value, ci in rows:
        lines.append(f"| {name} | {value:.9g} | [{ci[0]:.9g}, {ci[1]:.9g}] |")
        tex.append(f"{name} & {value:.6f} & [{ci[0]:.6f}, {ci[1]:.6f}] " + r"\\")
    tex += [r"\bottomrule", r"\end{tabularx}", r"\caption{" + caption + "}", r"\label{"+label+"}", r"\end{table}"]
    (root / "table.tex").write_text("\n".join(tex)+"\n")
    (root / "REPORT.md").write_text("\n".join(lines)+"\n\n"+caption+"\n")
    hashes(root)
