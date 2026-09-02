#!/usr/bin/env python3
"""Evaluate frozen V3 representations on ECtHR-A and 20 Newsgroups."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, f1_score, normalized_mutual_info_score,
                             precision_recall_fscore_support, accuracy_score)
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from recursive_geodesic_composition.architecture import GeodesicMerge  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
CACHE = ROOT / "results/v4_euclidean/downstream/leaf_cache"
OUTPUT = ROOT / "results/v4_euclidean/downstream/DOWNSTREAM_RESULTS.json"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def load_split(dataset: str, split: str, manifest: dict) -> dict:
    path = CACHE / dataset / f"{split}.pt"
    rel = str(path.relative_to(CACHE))
    expected = manifest["artifacts"][rel]["sha256"]
    if file_sha(path) != expected:
        raise ValueError(f"cache artifact hash mismatch: {rel}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if tensor_sha(data["leaves"]) != data["leaves_tensor_sha256"] or tensor_sha(data["lengths"]) != data["lengths_tensor_sha256"]:
        raise ValueError(f"cached tensor hash mismatch: {rel}")
    if len(data["ids"]) + 1 != len(data["offsets"]) or int(data["offsets"][-1]) != len(data["leaves"]):
        raise ValueError(f"cache offsets malformed: {rel}")
    return data


def averages(data: dict) -> tuple[np.ndarray, np.ndarray]:
    weighted, unweighted = [], []
    for a, b in zip(data["offsets"][:-1].tolist(), data["offsets"][1:].tolist()):
        leaves = data["leaves"][a:b].float()
        lengths = data["lengths"][a:b].float()
        # Canonical cleaned 20NG has a few zero-token rows. Their sole cached
        # leaf is the teacher's normalized CLS/SEP-only embedding; use it for
        # both means rather than dropping the row or dividing by zero.
        if float(lengths.sum()) == 0.0:
            if len(leaves) != 1:
                raise ValueError("zero-token row must have exactly one empty-sequence leaf")
            weighted.append(leaves[0])
        else:
            weighted.append(normalize((leaves * lengths[:, None]).sum(0)))
        unweighted.append(normalize(leaves.mean(0)))
    return torch.stack(weighted).numpy(), torch.stack(unweighted).numpy()


def load_model(entry: dict, device: torch.device) -> GeodesicMerge:
    path = ROOT / entry["path"]
    if file_sha(path) != entry["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {entry['name']}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg, state = ck["config"], ck["state_dict"]
    if int(ck["seed"]) != entry["seed"] or ck.get("selected_step") != 2400 or cfg.get("teacher_pooling") != "cls":
        raise ValueError(f"checkpoint provenance mismatch: {entry['name']}")
    d = int(state["input.0.weight"].shape[1] - 2)
    model = GeodesicMerge(
        d=d, hidden=int(cfg["encoder_hidden"]), layers=int(cfg["encoder_layers"]), heads=int(cfg["heads"]),
        max_angle=float(cfg["max_geodesic_angle"]), decoder=True, decoder_layers=int(cfg["decoder_layers"]),
        decoder_length_mode=str(cfg["decoder_length_mode"]), importance_mode=str(cfg["importance_mode"]),
        importance_beta=float(cfg["importance_beta"]), decoder_mode=str(cfg["decoder_mode"]),
        geodesic_parameterization=str(cfg["geodesic_parameterization"]), arity=4, composition_geometry=str(cfg["composition_geometry"]),
    ).to(device)
    model.load_state_dict(state, strict=True); model.eval()
    return model


@torch.inference_mode()
def recursive_all(model: GeodesicMerge, data: dict, device: torch.device, batch_size: int) -> np.ndarray:
    """Level-synchronous deterministic contiguous K=4 recursion, batched across documents."""
    nodes = []
    masses = []
    for a, b in zip(data["offsets"][:-1].tolist(), data["offsets"][1:].tolist()):
        nodes.append(data["leaves"][a:b].float())
        masses.append(data["lengths"][a:b].float())
    dtype = next(model.parameters()).dtype
    while any(len(x) > 1 for x in nodes):
        jobs, layouts = [], []
        for doc, (x, w) in enumerate(zip(nodes, masses)):
            layout = []
            for start in range(0, len(x), 4):
                width = min(4, len(x) - start)
                if width == 1:
                    layout.append(("carry", x[start], w[start]))
                else:
                    job_index = len(jobs)
                    jobs.append((x[start:start + width], w[start:start + width]))
                    layout.append(("merge", job_index))
            layouts.append(layout)
        outputs = [None] * len(jobs)
        for start in range(0, len(jobs), batch_size):
            part = jobs[start:start + batch_size]
            children = torch.stack([F.pad(x, (0, 0, 0, 4-len(x))) for x, _ in part]).to(device=device, dtype=dtype)
            weights = torch.stack([F.pad(w, (0, 4-len(w))) for _, w in part]).to(device=device, dtype=dtype)
            values = model.merge(children, weights).float().cpu()
            if not torch.isfinite(values).all():
                raise FloatingPointError("non-finite recursive embedding")
            for j, value in enumerate(values, start):
                outputs[j] = value
        new_nodes, new_masses = [], []
        for layout in layouts:
            doc_nodes, doc_masses = [], []
            for item in layout:
                if item[0] == "carry":
                    _, value, mass = item
                else:
                    value = outputs[item[1]]
                    x, w = jobs[item[1]]
                    mass = w.sum()
                doc_nodes.append(value); doc_masses.append(mass)
            new_nodes.append(torch.stack(doc_nodes)); new_masses.append(torch.stack(doc_masses))
        nodes, masses = new_nodes, new_masses
    result = torch.stack([x[0] for x in nodes]).numpy().astype(np.float32)
    if not np.isfinite(result).all() or len(result) != len(data["ids"]):
        raise FloatingPointError("invalid recursive result")
    return result


def multilabel_matrix(label_lists, width=11):
    y = np.zeros((len(label_lists), width), dtype=np.int8)
    for i, labels in enumerate(label_lists):
        y[i, labels] = 1
    if np.any(y.sum(1) != 1 + np.asarray([max(0, len(x)-1) for x in label_lists])):
        # Violation cases may have several labels; no-violation cases have exactly derived label 10.
        raise AssertionError("multilabel encoding failed")
    return y


def classification_metrics(y_true, y_pred, names):
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "violation_only_10_micro_f1": float(f1_score(y_true[:, :10], y_pred[:, :10], average="micro", zero_division=0)),
        "violation_only_10_macro_f1": float(f1_score(y_true[:, :10], y_pred[:, :10], average="macro", zero_division=0)),
        "per_label": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, name in enumerate(names)},
    }


def fit_ecthr(reps, labels, cfg):
    y_train, y_val, y_test = [multilabel_matrix(labels[s]) for s in ("train", "validation", "test")]
    results = {}
    for method, arrays in reps.items():
        validation = []
        for C in cfg["C_grid"]:
            estimator = OneVsRestClassifier(LogisticRegression(C=C, solver=cfg["solver"], max_iter=cfg["max_iter"], random_state=cfg["random_state"]))
            estimator.fit(arrays["train"], y_train)
            pred = estimator.predict(arrays["validation"])
            validation.append({"C": C, **classification_metrics(y_val, pred, [str(i) for i in range(11)])})
        best = max(validation, key=lambda x: (x["micro_f1"], -x["C"]))
        final = OneVsRestClassifier(LogisticRegression(C=best["C"], solver=cfg["solver"], max_iter=cfg["max_iter"], random_state=cfg["random_state"]))
        final.fit(np.concatenate([arrays["train"], arrays["validation"]]), np.concatenate([y_train, y_val]))
        test_pred = final.predict(arrays["test"])
        results[method] = {"selected_C": best["C"], "validation_grid": validation,
                           "test": classification_metrics(y_test, test_pred, ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1", "no_violation"])}
    return results


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--merge-batch-size", type=int, default=256)
    args = ap.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"create-only output exists: {OUTPUT}")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest_path = CACHE / "CACHE_MANIFEST.json"
    complete = CACHE / "COMPLETE"
    if not manifest_path.is_file() or complete.read_text().strip() != f"{file_sha(manifest_path)}  CACHE_MANIFEST.json":
        raise RuntimeError("leaf cache is absent or incomplete")
    manifest = json.loads(manifest_path.read_text())
    original_protocol = ROOT / "scripts/v3_cls/downstream/protocol.json"
    if manifest["protocol_sha256"] != file_sha(original_protocol):
        raise ValueError("reused teacher-only cache original protocol mismatch")
    if protocol.get("cache_reuse") is None:
        raise ValueError("V4 protocol must explicitly declare teacher-only cache reuse")
    started = time.perf_counter(); device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = {d: {s: load_split(d, s, manifest) for s in spec["splits"]} for d, spec in protocol["datasets"].items()}
    reps = {d: {s: {} for s in splits} for d, splits in data.items()}
    for d, splits in data.items():
        for s, value in splits.items():
            weighted, unweighted = averages(value)
            reps[d][s]["token_weighted_mean"] = weighted
            reps[d][s]["unweighted_mean"] = unweighted
    recursive_seconds = {}
    for checkpoint in protocol["checkpoints"]:
        tick = time.perf_counter(); model = load_model(checkpoint, device)
        for d, splits in data.items():
            for s, value in splits.items():
                reps[d][s][checkpoint["name"]] = recursive_all(model, value, device, args.merge_batch_size)
        recursive_seconds[checkpoint["name"]] = time.perf_counter() - tick
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    direct_coverage = {}
    for d, splits in data.items():
        direct_coverage[d] = {}
        all_valid = all(bool(value["direct_valid"].all()) for value in splits.values())
        for s, value in splits.items():
            direct_coverage[d][s] = {"valid": int(value["direct_valid"].sum()), "invalid": int((~value["direct_valid"]).sum()), "rows": len(value["ids"])}
            if all_valid:
                reps[d][s]["direct_teacher"] = value["direct_teacher"].numpy()
        direct_coverage[d]["included_in_task"] = all_valid
        direct_coverage[d]["exclusion_reason"] = None if all_valid else "at least one task row exceeds pinned teacher context; identical-sample policy forbids subset evaluation"
    # Transpose split->method into method->split and assert identical row counts.
    ecthr_reps = {m: {s: reps["ecthr_a"][s][m] for s in ("train", "validation", "test")} for m in reps["ecthr_a"]["train"]}
    labels = {s: data["ecthr_a"][s]["labels"] for s in ("train", "validation", "test")}
    class_started = time.perf_counter()
    ecthr = fit_ecthr(ecthr_reps, labels, protocol["classification"])
    class_seconds = time.perf_counter() - class_started
    clustering = {}
    cluster_cfg = protocol["clustering"]
    for method in reps["20_newsgroups"]["train"]:
        x = np.concatenate([reps["20_newsgroups"][s][method] for s in ("train", "test")])
        y = np.asarray(data["20_newsgroups"]["train"]["labels"] + data["20_newsgroups"]["test"]["labels"])
        km = KMeans(n_clusters=cluster_cfg["n_clusters"], init=cluster_cfg["init"], n_init=cluster_cfg["n_init"],
                    max_iter=cluster_cfg["max_iter"], algorithm=cluster_cfg["algorithm"], random_state=cluster_cfg["random_state"])
        tick = time.perf_counter(); predicted = km.fit_predict(x)
        clustering[method] = {"rows": len(y), "nmi": float(normalized_mutual_info_score(y, predicted, average_method="arithmetic")),
                              "ari": float(adjusted_rand_score(y, predicted)), "inertia": float(km.inertia_),
                              "iterations": int(km.n_iter_), "seconds": time.perf_counter() - tick}
    result = {
        "schema": "v4_euclidean_downstream_results_v1", "status": "complete", "ecthr_a": ecthr, "20_newsgroups": clustering,
        "direct_teacher_coverage": direct_coverage,
        "runtime": {"total_seconds": time.perf_counter()-started, "recursive_seconds": recursive_seconds, "classification_seconds": class_seconds},
        "provenance": {"protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "protocol_sha256": file_sha(PROTOCOL_PATH),
                       "cache_manifest": str(manifest_path.relative_to(ROOT)), "cache_manifest_sha256": file_sha(manifest_path),
                       "evaluator": str(Path(__file__).relative_to(ROOT)), "evaluator_sha256": file_sha(Path(__file__)),
                       "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                       "sklearn": sklearn.__version__, "device": str(device), "checkpoints": protocol["checkpoints"]},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    (OUTPUT.parent / "DOWNSTREAM_COMPLETE").write_text(f"{file_sha(OUTPUT)}  {OUTPUT.name}\n")
    print(OUTPUT)

if __name__ == "__main__":
    main()
