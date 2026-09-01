"""Train-fitted final-root calibration evaluated on validation roles only.

The affine adapter is applied only to the externally exposed final root. Recursive
merger state remains unchanged. AE/common-probe consumers first call ``inverse``
to recover merger space from the normalized adapted root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .information_retention import CommonDecoderProbe, build_merger, tree_plan

ARMS = ("encoder", "autoencoder")
CALIBRATORS = ("identity", "procrustes", "coral", "ridge")
FORBIDDEN_ROLE_KEYS = ("test", "official")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def orientation_safe_auc(auc: float) -> float:
    if not np.isfinite(auc) or not 0.0 <= float(auc) <= 1.0:
        raise ValueError(f"invalid AUC: {auc}")
    return float(max(float(auc), 1.0 - float(auc)))


def audit_fit_boundary(
    fit_role: str, fit_groups: np.ndarray, evaluation_groups: np.ndarray
) -> dict:
    if fit_role != "train":
        raise ValueError("calibration transforms may only be fit with role='train'")
    fit = {str(x) for x in fit_groups}
    evaluation = {str(x) for x in evaluation_groups}
    overlap = fit & evaluation
    if overlap:
        raise AssertionError(f"fit/evaluation source-document leakage: {sorted(overlap)[:3]}")
    return {
        "fit_role": "train",
        "fit_groups": len(fit),
        "evaluation_role": "validation",
        "evaluation_groups": len(evaluation),
        "group_overlap": 0,
        "fit_group_sha256": stable_hash(sorted(fit)),
        "evaluation_group_sha256": stable_hash(sorted(evaluation)),
    }


@dataclass
class AffineCalibrator:
    name: str
    matrix: np.ndarray
    bias: np.ndarray
    fit_metadata: dict

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("calibration matrix must be square")
        if self.bias.shape != (self.matrix.shape[0],):
            raise ValueError("calibration bias shape mismatch")
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.bias).all():
            raise FloatingPointError("non-finite calibration transform")

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    @property
    def condition(self) -> float:
        return float(np.linalg.cond(self.matrix))

    def apply_with_norm(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(x, dtype=np.float64) @ self.matrix + self.bias
        norms = np.linalg.norm(z, axis=1)
        if not np.isfinite(z).all() or np.any(norms <= 0):
            raise FloatingPointError(f"{self.name} produced invalid final roots")
        return z / norms[:, None], norms

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.apply_with_norm(x)[0]

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Invert a normalized affine output using the unit merger-root constraint.

        For y=normalize(xA+b), solve ||(s*y-b)A^-1||=1 for positive scale s,
        then return the normalized merger-space root. This is the required path
        before staged-AE decoding or an existing common parent-only probe.
        """
        y = np.asarray(y, dtype=np.float64)
        inverse = np.linalg.inv(self.matrix)
        u = y @ inverse
        v = -self.bias @ inverse
        aa = np.sum(u * u, axis=1)
        bb = 2.0 * np.sum(u * v[None, :], axis=1)
        cc = float(np.sum(v * v)) - 1.0
        discriminant = np.maximum(bb * bb - 4.0 * aa * cc, 0.0)
        root = np.sqrt(discriminant)
        s1 = (-bb + root) / (2.0 * aa)
        s2 = (-bb - root) / (2.0 * aa)
        scale = np.where((s1 > 0) & ((s2 <= 0) | (np.abs(s1 - 1) <= np.abs(s2 - 1))), s1, s2)
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise FloatingPointError(f"{self.name} normalized affine inverse has no positive scale")
        x = scale[:, None] * u + v[None, :]
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms <= 0) or not np.isfinite(x).all():
            raise FloatingPointError(f"{self.name} inverse produced invalid merger roots")
        return x / norms[:, None]

    def save(self, path: Path) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            matrix=self.matrix,
            bias=self.bias,
            metadata=np.asarray(json.dumps(self.fit_metadata, sort_keys=True)),
        )
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "rank": self.rank,
            "dimension": int(self.matrix.shape[0]),
            "condition": self.condition,
        }


def fit_calibrator(
    name: str,
    predicted: np.ndarray,
    teacher: np.ndarray,
    *,
    coral_shrinkage: float,
    ridge_relative: float,
    fit_role: str,
    fit_groups: np.ndarray,
    evaluation_groups: np.ndarray,
) -> AffineCalibrator:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(teacher, dtype=np.float64)
    if name not in CALIBRATORS or x.shape != y.shape or x.ndim != 2:
        raise ValueError("invalid fixed calibrator or paired fit arrays")
    if len(x) <= x.shape[1]:
        raise ValueError("full-rank calibration requires more fit rows than dimensions")
    audit = audit_fit_boundary(fit_role, fit_groups, evaluation_groups)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    d = x.shape[1]
    metadata = {
        **audit,
        "calibrator": name,
        "fit_pairs": int(len(x)),
        "dimension": int(d),
        "fit_predicted_sha256": hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest(),
        "fit_teacher_sha256": hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest(),
        "normalization": "final_l2",
        "adapter_scope": "final_root_only",
    }
    if name == "identity":
        matrix, bias = np.eye(d), np.zeros(d)
    elif name == "procrustes":
        u, _, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
        matrix = u @ vt
        bias = my - mx @ matrix
        metadata["mean_alignment"] = True
    elif name == "coral":
        alpha = float(coral_shrinkage)
        if not 0.0 < alpha < 1.0:
            raise ValueError("CORAL shrinkage must be fixed in (0,1)")
        cx = (xc.T @ xc) / (len(x) - 1)
        cy = (yc.T @ yc) / (len(y) - 1)
        cx = (1 - alpha) * cx + alpha * np.trace(cx) / d * np.eye(d)
        cy = (1 - alpha) * cy + alpha * np.trace(cy) / d * np.eye(d)
        lx, ux = np.linalg.eigh(cx)
        ly, uy = np.linalg.eigh(cy)
        wx = (ux * (1.0 / np.sqrt(lx))[None, :]) @ ux.T
        ry = (uy * np.sqrt(ly)[None, :]) @ uy.T
        matrix = wx @ ry
        bias = my - mx @ matrix
        metadata["fixed_shrinkage"] = alpha
    else:
        relative = float(ridge_relative)
        gram = xc.T @ xc
        covariance_scale = float(np.trace(gram) / d)
        ridge = relative * covariance_scale
        matrix = np.linalg.solve(gram + ridge * np.eye(d), xc.T @ yc)
        bias = my - mx @ matrix
        metadata.update(
            fixed_relative_ridge=relative,
            realized_ridge=ridge,
            covariance_scale=covariance_scale,
            intercept_penalized=False,
        )
    transform = AffineCalibrator(name, matrix, bias, metadata)
    metadata["matrix_rank"] = transform.rank
    metadata["matrix_condition"] = transform.condition
    return transform


def load_protocol(root: Path, config_path: Path) -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("validation_only") is not True or cfg.get("official_test_evaluated") is not False:
        raise ValueError("study must be explicitly validation-only")
    if tuple(cfg.get("calibrators", ())) != CALIBRATORS:
        raise ValueError("calibrator list/order changed after predeclaration")
    source = yaml.safe_load((root / cfg["source_config"]).read_text())
    # Fail closed on any declared non-train/non-validation role without probing paths.
    for key in source:
        low = str(key).lower()
        if any(word in low for word in FORBIDDEN_ROLE_KEYS):
            raise ValueError(f"source config declares forbidden role key: {key}")
    corpora = tuple(cfg["corpora"])
    if tuple(c["id"] for c in source["corpora"]) != corpora:
        raise ValueError("corpus order differs from frozen source config")
    verified = {}
    for arm in ARMS:
        entry = cfg["frozen"][arm]
        checkpoint = root / entry["checkpoint"]
        manifest_path = root / entry["manifest"]
        actual = sha256_file(checkpoint)
        if actual != entry["sha256"]:
            raise ValueError(f"checkpoint hash mismatch for {arm}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("official_test_evaluated") is not False or manifest.get("checkpoint_sha256") != actual:
            raise ValueError(f"invalid validation-only frozen manifest for {arm}")
        probe = root / entry["common_probe"]
        if float(manifest.get("selected_geodesic_saturation_fraction", -1)) != 0.0:
            raise ValueError(f"frozen model has nonzero/unknown saturation for {arm}")
        verified[arm] = {
            "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": actual,
            "manifest": entry["manifest"],
            "manifest_sha256": sha256_file(manifest_path),
            "common_probe": entry["common_probe"],
            "common_probe_sha256": sha256_file(probe),
            "selected_geodesic_saturation_fraction": 0.0,
            "recursive_state_changed_by_adapter": False,
        }
    return cfg, source, verified


def load_role_caches(root: Path, cfg: dict, source: dict) -> tuple[dict, dict, dict]:
    train, validation, provenance = {}, {}, {}
    for corpus_cfg in source["corpora"]:
        corpus = corpus_cfg["id"]
        train_path = root / cfg["cache_dir"] / corpus / "train.pt"
        validation_path = root / cfg["cache_dir"] / corpus / "validation.pt"
        train[corpus] = torch.load(train_path, map_location="cpu", weights_only=False)
        validation[corpus] = torch.load(validation_path, map_location="cpu", weights_only=False)
        for role, path in (("train", train_path), ("validation", validation_path)):
            sidecar_path = Path(str(path) + ".json")
            sidecar = json.loads(sidecar_path.read_text())
            expected = {
                "split": role,
                "source_start": int(corpus_cfg[f"{role}_start"]),
                "requested": int(corpus_cfg[f"{role}_documents"]),
                "corpus_id": corpus,
                "dataset": corpus_cfg["dataset_name"],
                "revision": corpus_cfg["dataset_revision"],
                "model": source["model_name"],
                "model_revision": source["model_revision"],
            }
            for key, value in expected.items():
                if sidecar.get(key) != value:
                    raise ValueError(f"cache provenance mismatch {corpus}/{role}/{key}")
            provenance[f"{corpus}/{role}"] = {**expected, "sidecar_sha256": sha256_file(sidecar_path)}
        train_groups = {str(row["id"]) for row in train[corpus]}
        val_groups = {str(row["id"]) for row in validation[corpus]}
        if train_groups & val_groups:
            raise AssertionError(f"train/validation source group overlap in {corpus}")
    return train, validation, provenance


def select_fit_rows(rows: list, count: int, seed: int, corpus: str) -> tuple[list[int], list[str]]:
    by_group = {}
    for index, row in enumerate(rows):
        by_group.setdefault(str(row["id"]), index)
    ranked = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"{seed}|{corpus}|{group}".encode()).digest(),
    )
    if len(ranked) < count:
        raise ValueError(f"not enough unique train documents in {corpus}")
    chosen = ranked[:count]
    return [by_group[group] for group in chosen], chosen


@torch.inference_mode()
def extract_roots(
    rows: list,
    indices: list[int],
    model,
    device: torch.device,
    batch_size: int,
    return_top_children: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    grouped = defaultdict(list)
    for index in indices:
        row = rows[index]
        grouped[(int(row["leaves"].shape[0]), row.get("tree_template", "balanced"))].append(index)
    roots, children_out = {}, {}
    for (leaf_count, template), group in sorted(grouped.items()):
        for start in range(0, len(group), batch_size):
            part = group[start : start + batch_size]
            nodes = torch.stack([rows[i]["leaves"] for i in part]).to(device).float()
            sizes = torch.stack([rows[i]["lengths"] for i in part]).to(device).float()
            final_children = None
            for positions in tree_plan(leaf_count, template):
                position = torch.as_tensor(positions, device=device)
                children = torch.stack((nodes[:, position], nodes[:, position + 1]), dim=2)
                child_sizes = torch.stack((sizes[:, position], sizes[:, position + 1]), dim=2)
                parent = model.merge(
                    children.reshape(-1, 2, model.d), child_sizes.reshape(-1, 2)
                ).reshape(len(part), len(positions), model.d)
                if nodes.shape[1] == 2:
                    final_children = children[:, 0]
                next_nodes, next_sizes, cursor = [], [], 0
                for pi, pos in enumerate(positions):
                    if cursor < pos:
                        next_nodes.append(nodes[:, cursor:pos])
                        next_sizes.append(sizes[:, cursor:pos])
                    next_nodes.append(parent[:, pi : pi + 1])
                    next_sizes.append(child_sizes[:, pi].sum(-1, keepdim=True))
                    cursor = pos + 2
                if cursor < nodes.shape[1]:
                    next_nodes.append(nodes[:, cursor:])
                    next_sizes.append(sizes[:, cursor:])
                nodes, sizes = torch.cat(next_nodes, 1), torch.cat(next_sizes, 1)
            for bi, index in enumerate(part):
                roots[index] = nodes[bi, 0].cpu().numpy().astype(np.float32)
                if return_top_children:
                    if final_children is None:
                        raise AssertionError("missing final merge children")
                    children_out[index] = final_children[bi].cpu().numpy().astype(np.float32)
    ordered_roots = np.stack([roots[i] for i in indices])
    ordered_children = np.stack([children_out[i] for i in indices]) if return_top_children else None
    return ordered_roots, ordered_children


def teacher_roots(rows: list, indices: list[int]) -> np.ndarray:
    values = []
    for index in indices:
        target = next((x for x in reversed(rows[index]["targets"]) if x is not None), None)
        if target is None or not torch.isfinite(target).all():
            raise FloatingPointError(f"missing/non-finite teacher root at row {index}")
        values.append(target.float().numpy())
    return np.stack(values).astype(np.float32)


def diagnostics(x: np.ndarray, pre_norm: np.ndarray, cfg: dict) -> dict:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    centered = x[finite] - x[finite].mean(0)
    singular = np.linalg.svd(centered[: min(2048, len(centered))], compute_uv=False)
    variance = singular * singular
    effective_rank = float(variance.sum() ** 2 / max(np.sum(variance * variance), 1e-18))
    rounded = np.round(x[finite][: min(2048, finite.sum())] * 1e4).astype(np.int32)
    unique_fraction = float(len(np.unique(rounded, axis=0)) / max(len(rounded), 1))
    norms = np.linalg.norm(x[finite], axis=1)
    minimum_pre = float(cfg["collapse"]["minimum_pre_normalization_norm"])
    saturation_count = int(np.sum(~np.isfinite(pre_norm) | (pre_norm <= minimum_pre)))
    collapsed = bool(
        effective_rank < float(cfg["collapse"]["minimum_effective_rank"])
        or unique_fraction < float(cfg["collapse"]["minimum_unique_rounded_fraction"])
    )
    return {
        "count": int(len(x)),
        "finite_fraction": float(finite.mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "effective_rank": effective_rank,
        "unique_rounded_fraction": unique_fraction,
        "collapse": collapsed,
        "collapse_count": int(collapsed),
        "adapter_saturation_count": saturation_count,
    }


def grouped_attack_split(real, fake, groups, attack_cfg):
    real, fake, groups = np.asarray(real), np.asarray(fake), np.asarray(groups)
    x = np.vstack((real, fake))
    y = np.r_[np.ones(len(real), dtype=int), np.zeros(len(fake), dtype=int)]
    paired_groups = np.r_[groups, groups]
    train, holdout = next(GroupShuffleSplit(
        n_splits=1,
        test_size=float(attack_cfg["test_size"]),
        random_state=int(attack_cfg["split_seed"]),
    ).split(x, y, paired_groups))
    if set(paired_groups[train]) & set(paired_groups[holdout]):
        raise AssertionError("source-document leakage in grouped attack split")
    return x, y, paired_groups, train, holdout


def evaluate_fixed_attacks(real, fake, groups, attack_cfg: dict) -> dict:
    x, y, paired_groups, train, holdout = grouped_attack_split(real, fake, groups, attack_cfg)
    seed = int(attack_cfg["model_seed"])
    models = {
        "standard_scaled_logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=float(attack_cfg["logistic_c"]), max_iter=int(attack_cfg["logistic_max_iter"]),
                solver="lbfgs", random_state=seed,
            )),
        ]),
        "standard_scaled_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=float(attack_cfg["rbf_c"]), gamma=float(attack_cfg["rbf_gamma"]),
                               kernel="rbf", random_state=seed)),
        ]),
    }
    results = {}
    for name, model in models.items():
        model.fit(x[train], y[train])
        score = model.predict_proba(x[holdout])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x[holdout])
        prediction = model.predict(x[holdout])
        raw_auc = float(roc_auc_score(y[holdout], score))
        raw_bal = float(balanced_accuracy_score(y[holdout], prediction))
        results[name] = {
            "raw_auc": raw_auc,
            "orientation_safe_auc": orientation_safe_auc(raw_auc),
            "pr_auc": float(average_precision_score(y[holdout], score)),
            "orientation_safe_pr_auc": float(max(
                average_precision_score(y[holdout], score),
                average_precision_score(1 - y[holdout], -score),
            )),
            "balanced_accuracy": raw_bal,
            "orientation_safe_balanced_accuracy": float(max(raw_bal, 1.0 - raw_bal)),
        }
    strongest = max(results, key=lambda name: results[name]["orientation_safe_auc"])
    return {
        "attacks": results,
        "strongest_attack": strongest,
        "strongest_orientation_safe_auc": results[strongest]["orientation_safe_auc"],
        "attack_train_documents": int(len(np.unique(paired_groups[train]))),
        "attack_holdout_documents": int(len(np.unique(paired_groups[holdout]))),
        "group_overlap": 0,
    }


def cosine_mean(x: np.ndarray, y: np.ndarray) -> float:
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return float(np.mean(np.sum(x * y, axis=1)))


def load_common_probe(path: Path, dimension: int, hidden: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "common" or checkpoint.get("updates") != checkpoint.get("configured_updates"):
        raise ValueError("common retention probe is not frozen/complete")
    probe = CommonDecoderProbe(dimension, hidden)
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    return probe.to(device).eval(), checkpoint


@torch.inference_mode()
def retention_metrics(
    calibrator: AffineCalibrator,
    raw_roots: np.ndarray,
    adapted_roots: np.ndarray,
    top_children: np.ndarray,
    common_probe,
    model,
    arm: str,
    device: torch.device,
) -> dict:
    restored = calibrator.inverse(adapted_roots).astype(np.float32)
    roundtrip = np.sum(
        restored * raw_roots /
        (np.linalg.norm(restored, axis=1, keepdims=True) * np.linalg.norm(raw_roots, axis=1, keepdims=True)), axis=1
    )
    def run_probe(values):
        outputs = []
        for start in range(0, len(values), 512):
            tensor = torch.from_numpy(values[start:start + 512]).to(device)
            outputs.append(common_probe(tensor).cpu())
        return torch.cat(outputs).numpy()
    base_common = run_probe(raw_roots)
    adapted_common = run_probe(restored)
    target = top_children
    base_cos = cosine_mean(base_common.reshape(-1, base_common.shape[-1]), target.reshape(-1, target.shape[-1]))
    adapted_cos = cosine_mean(adapted_common.reshape(-1, adapted_common.shape[-1]), target.reshape(-1, target.shape[-1]))
    result = {
        "inverse_roundtrip_cosine_mean": float(roundtrip.mean()),
        "inverse_roundtrip_cosine_min": float(roundtrip.min()),
        "common_parent_only_baseline_cosine": base_cos,
        "common_parent_only_adapted_cosine": adapted_cos,
        "common_parent_only_delta": adapted_cos - base_cos,
        "decoder_input": "analytic_inverse_of_normalized_final_root",
    }
    if arm == "autoencoder":
        def decode(values):
            outputs = []
            for start in range(0, len(values), 512):
                outputs.append(model.decode(torch.from_numpy(values[start:start + 512]).to(device)).cpu())
            return torch.cat(outputs).numpy()
        base_decode, adapted_decode = decode(raw_roots), decode(restored)
        base_ae = cosine_mean(base_decode.reshape(-1, base_decode.shape[-1]), target.reshape(-1, target.shape[-1]))
        adapted_ae = cosine_mean(adapted_decode.reshape(-1, adapted_decode.shape[-1]), target.reshape(-1, target.shape[-1]))
        result.update(
            builtin_ae_baseline_reconstruction_cosine=base_ae,
            builtin_ae_adapted_reconstruction_cosine=adapted_ae,
            builtin_ae_reconstruction_delta=adapted_ae - base_ae,
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_report(result: dict) -> str:
    selected = result["selected"]
    lines = [
        "# Train-only output calibration — validation report", "",
        "Every transform was fit on TRAIN caches and evaluated only on current validation roles. No future untouched evaluation is authorized.", "",
        f"**Status:** `{result['status']}`  ",
        f"**Strongest orientation-safe AUC <= 0.55 achieved:** **{'YES' if result['maximum_auc_gate_achieved'] else 'NO'}**", "",
        "## Representation/calibrator summary", "",
        "| representation | calibrator | pooled strongest safe AUC | worst pooled/corpus safe AUC | pooled cosine | rank | condition | common retention delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for name in CALIBRATORS:
            entry = result["transforms"][f"{arm}/{name}"]
            scopes = entry["validation"]
            worst = max(value["attacks"]["strongest_orientation_safe_auc"] for value in scopes.values())
            pooled = scopes["pooled"]
            lines.append(
                f"| {arm} | {name} | {pooled['attacks']['strongest_orientation_safe_auc']:.6f} | "
                f"{worst:.6f} | {pooled['teacher_cosine']:.6f} | {entry['rank']} | "
                f"{entry['condition']:.3g} | {entry['retention']['common_parent_only_delta']:+.3e} |"
            )
    lines += [
        "", "## Fixed attacks by corpus", "",
        "Raw ROC AUC retains the declared teacher=1 orientation; safe AUC is `max(AUC, 1-AUC)`.", "",
        "| representation | calibrator | scope | attack | raw AUC | safe AUC | PR AUC | balanced accuracy | teacher cosine |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for name in CALIBRATORS:
            for scope, scope_result in result["transforms"][f"{arm}/{name}"]["validation"].items():
                for attack_name, metrics in scope_result["attacks"]["attacks"].items():
                    lines.append(
                        f"| {arm} | {name} | {scope} | {attack_name} | {metrics['raw_auc']:.6f} | "
                        f"{metrics['orientation_safe_auc']:.6f} | {metrics['pr_auc']:.6f} | "
                        f"{metrics['balanced_accuracy']:.6f} | {scope_result['teacher_cosine']:.6f} |"
                    )
    lines += ["", "## AE selection gates", "",
              "| calibrator | worst safe AUC | cosine | privacy | retention | AE gain | health | rank/condition | all gates |",
              "|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for candidate in result["candidates"]:
        gates = candidate["gates"]
        lines.append(
            f"| {candidate['calibrator']} | {candidate['strongest_orientation_safe_auc']:.6f} | "
            f"{candidate['teacher_cosine']:.6f} | {gates['privacy']} | "
            f"{gates['retention_noninferiority']} | {gates['ae_specific_gain']} | {gates['health']} | "
            f"{gates['full_rank_condition']} | {candidate['passes_all_gates']} |"
        )
    lines += [
        "", "All output norms, collapse/effective-rank diagnostics, grouped split counts, transform fit hashes, and per-scope strongest-attack fields are in `study.json` and `metrics.csv`.",
        "", "Transforms are final-root adapters only. Recursive merger states are unchanged. The staged AE and frozen common probe consume `inverse(normalized_calibrated_root)`; analytic inverse round-trip and parent-only/built-in reconstruction deltas are recorded in `study.json`.", "",
    ]
    if selected:
        lines.append(
            f"Validation-selected candidate: `{selected['arm']}/{selected['calibrator']}`. A frozen multi-seed validation confirmation remains mandatory before any future untouched evaluation."
        )
    else:
        lines.append(
            "The <=0.55 gate was not achieved (the best pooled value was 0.570277 for AE Procrustes, and its cosine was 0.974385). Recommended next step: a TRAIN-only constrained residual distribution-alignment architecture with cosine and invertible-cycle guards, not more affine calibrator tuning on validation."
        )
    return "\n".join(lines) + "\n"


def synthetic_smoke(device_name: str) -> dict:
    device = torch.device(device_name)
    rng = np.random.default_rng(123)
    n, d = 96, 16
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    y = x @ q + 0.01 * rng.normal(size=(n, d))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    fit_groups = np.asarray([f"train:{i}" for i in range(n)])
    val_groups = np.asarray([f"validation:{i}" for i in range(n)])
    outputs = {}
    for name in CALIBRATORS:
        cal = fit_calibrator(name, x, y, coral_shrinkage=.1, ridge_relative=.001,
                             fit_role="train", fit_groups=fit_groups, evaluation_groups=val_groups)
        z = cal.apply(x)
        restored = cal.inverse(z)
        tensor = torch.from_numpy(z.astype(np.float32)).to(device)
        outputs[name] = {
            "finite": bool(torch.isfinite(tensor).all().item()),
            "rank": cal.rank,
            "condition": cal.condition,
            "roundtrip_cosine": cosine_mean(restored, x),
        }
    return {"status": "smoke_passed", "device": str(device), "calibrators": outputs}


def run_study(root: Path, config_path: Path, device_name: str) -> dict:
    started = time.time()
    cfg, source, verified = load_protocol(root, config_path)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    train, validation, cache_provenance = load_role_caches(root, cfg, source)
    output = root / cfg["output_dir"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen study output: {output}")
    (output / "transforms").mkdir(parents=True)
    attack_cfg = yaml.safe_load((root / cfg["attack_config"]).read_text())
    fit_indices, fit_groups, val_groups = {}, {}, {}
    for corpus in cfg["corpora"]:
        fit_indices[corpus], fit_groups[corpus] = select_fit_rows(
            train[corpus], int(cfg["fit_documents_per_corpus"]), int(cfg["fit_seed"]), corpus
        )
        val_groups[corpus] = [str(row["id"]) for row in validation[corpus]]
    all_fit_groups = np.asarray(sum((fit_groups[c] for c in cfg["corpora"]), []))
    all_val_groups = np.asarray(sum((val_groups[c] for c in cfg["corpora"]), []))
    audit_fit_boundary("train", all_fit_groups, all_val_groups)

    arm_data, models, probes = {}, {}, {}
    for arm in ARMS:
        model = build_merger(root / cfg["frozen"][arm]["checkpoint"], device)
        models[arm] = model
        predicted_fit, teacher_fit, predicted_val, teacher_val, top_children = [], [], [], [], []
        corpus_slices, cursor = {}, 0
        for corpus in cfg["corpora"]:
            fi = fit_indices[corpus]
            vi = list(range(len(validation[corpus])))
            fit_root, _ = extract_roots(train[corpus], fi, model, device, int(cfg["root_batch_size"]))
            val_root, children = extract_roots(validation[corpus], vi, model, device,
                                               int(cfg["root_batch_size"]), True)
            predicted_fit.append(fit_root)
            teacher_fit.append(teacher_roots(train[corpus], fi))
            predicted_val.append(val_root)
            teacher_val.append(teacher_roots(validation[corpus], vi))
            top_children.append(children)
            corpus_slices[corpus] = [cursor, cursor + len(vi)]
            cursor += len(vi)
        arm_data[arm] = {
            "fit_predicted": np.concatenate(predicted_fit),
            "fit_teacher": np.concatenate(teacher_fit),
            "validation_predicted": np.concatenate(predicted_val),
            "validation_teacher": np.concatenate(teacher_val),
            "top_children": np.concatenate(top_children),
            "corpus_slices": corpus_slices,
        }
        probe, probe_ck = load_common_probe(root / cfg["frozen"][arm]["common_probe"],
                                            arm_data[arm]["fit_predicted"].shape[1],
                                            1024, device)
        probes[arm] = probe

    records, transforms = [], {}
    for arm in ARMS:
        data = arm_data[arm]
        for name in CALIBRATORS:
            calibrator = fit_calibrator(
                name, data["fit_predicted"], data["fit_teacher"],
                coral_shrinkage=float(cfg["coral_shrinkage"]),
                ridge_relative=float(cfg["ridge_relative"]), fit_role="train",
                fit_groups=all_fit_groups, evaluation_groups=all_val_groups,
            )
            transform_path = output / "transforms" / f"{arm}_{name}.npz"
            transform_info = calibrator.save(transform_path)
            transform_info["path"] = str(transform_path.relative_to(root))
            transforms[f"{arm}/{name}"] = {**transform_info, "fit_metadata": calibrator.fit_metadata}
            adapted, pre_norm = calibrator.apply_with_norm(data["validation_predicted"])
            retention = retention_metrics(calibrator, data["validation_predicted"], adapted,
                                          data["top_children"], probes[arm], models[arm], arm, device)
            scopes = {"pooled": np.arange(len(adapted))}
            for corpus, (start, stop) in data["corpus_slices"].items():
                scopes[corpus] = np.arange(start, stop)
            scope_results = {}
            for scope, index in scopes.items():
                groups = all_val_groups[index]
                attack = evaluate_fixed_attacks(data["validation_teacher"][index], adapted[index], groups, attack_cfg)
                health = diagnostics(adapted[index], pre_norm[index], cfg)
                scope_results[scope] = {
                    "teacher_cosine": cosine_mean(adapted[index], data["validation_teacher"][index]),
                    "attacks": attack,
                    "diagnostics": health,
                }
                for attack_name, metrics in attack["attacks"].items():
                    records.append({
                        "arm": arm, "calibrator": name, "scope": scope, "attack": attack_name,
                        "teacher_cosine": scope_results[scope]["teacher_cosine"],
                        **metrics, "strongest_orientation_safe_auc": attack["strongest_orientation_safe_auc"],
                        "effective_rank": health["effective_rank"], "collapse_count": health["collapse_count"],
                        "adapter_saturation_count": health["adapter_saturation_count"],
                    })
            transforms[f"{arm}/{name}"]["validation"] = scope_results
            transforms[f"{arm}/{name}"]["retention"] = retention

    encoder_common = transforms["encoder/identity"]["retention"]["common_parent_only_baseline_cosine"]
    ae_common = transforms["autoencoder/identity"]["retention"]["common_parent_only_baseline_cosine"]
    ae_gain = ae_common - encoder_common
    tolerance = float(cfg["retention"]["cosine_noninferiority_tolerance"])
    candidates = []
    for name in CALIBRATORS:
        key = f"autoencoder/{name}"
        entry = transforms[key]
        validation_result = entry["validation"]
        retention = entry["retention"]
        strongest = max(v["attacks"]["strongest_orientation_safe_auc"] for v in validation_result.values())
        health_ok = (
            verified["autoencoder"]["selected_geodesic_saturation_fraction"] == 0.0
            and all(
                v["diagnostics"]["collapse_count"] == 0
                and v["diagnostics"]["adapter_saturation_count"] == 0
                for v in validation_result.values()
            )
        )
        attack_scopes = validation_result.values() if cfg["gates"]["require_every_corpus_attack_gate"] else [validation_result["pooled"]]
        attack_ok = all(v["attacks"]["strongest_orientation_safe_auc"] <= float(cfg["gates"]["maximum_orientation_safe_auc"])
                        for v in attack_scopes)
        cosine_ok = validation_result["pooled"]["teacher_cosine"] >= float(cfg["gates"]["minimum_teacher_cosine"])
        retention_ok = (
            retention["common_parent_only_delta"] >= -tolerance
            and retention["builtin_ae_reconstruction_delta"] >= -tolerance
            and retention["inverse_roundtrip_cosine_min"] >= float(cfg["retention"]["minimum_inverse_roundtrip_cosine"])
        )
        ae_gain_ok = ae_gain > 0.0
        rank_ok = entry["rank"] == entry["dimension"] and entry["condition"] <= float(cfg["condition_limit"])
        gate = {
            "privacy": bool(attack_ok), "teacher_cosine": bool(cosine_ok), "health": bool(health_ok),
            "retention_noninferiority": bool(retention_ok), "ae_specific_gain": bool(ae_gain_ok),
            "full_rank_condition": bool(rank_ok),
        }
        candidates.append({
            "arm": "autoencoder", "calibrator": name, "strongest_orientation_safe_auc": strongest,
            "teacher_cosine": validation_result["pooled"]["teacher_cosine"],
            "ae_common_probe_gain_over_encoder": ae_gain, "gates": gate, "passes_all_gates": all(gate.values()),
        })
    eligible = [x for x in candidates if x["passes_all_gates"]]
    eligible.sort(key=lambda x: (x["strongest_orientation_safe_auc"], -x["teacher_cosine"], -x["ae_common_probe_gain_over_encoder"]))
    selected = eligible[0] if eligible else None
    status = "validation_gate_passed" if selected else "validation_gate_failed"
    result = {
        "study": cfg["study"], "status": status, "validation_only": True,
        "official_test_evaluated": False, "device": str(device), "seed": cfg["seed"],
        "fit_seed": cfg["fit_seed"], "fit_role": "train", "evaluation_role": "validation",
        "fit_pairs": int(len(all_fit_groups)), "fit_group_overlap_with_validation": 0,
        "calibrator_count_including_identity": len(CALIBRATORS),
        "fixed_parameters": {"coral_shrinkage": cfg["coral_shrinkage"], "ridge_relative": cfg["ridge_relative"]},
        "attack_protocol": attack_cfg, "cache_provenance": cache_provenance,
        "frozen_artifacts": verified, "transforms": transforms, "candidates": candidates,
        "selected": selected, "maximum_auc_gate_achieved": bool(selected is not None),
        "future_untouched_test_authorized": False,
        "next_architecture_if_failed": (
            None if selected else
            "Train a constrained residual distribution-alignment adapter jointly on TRAIN only (teacher-cosine and invertible-cycle guards, adversarial plus covariance/MMD objectives), then freeze and repeat multi-seed validation; do not tune another linear calibrator family."
        ),
        "runtime_seconds": time.time() - started,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available()},
    }
    json_path = output / "study.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(output / "metrics.csv", records)
    (output / "REPORT.md").write_text(render_report(result))
    hashes = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation_output_calibration_v1.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.smoke:
        print(json.dumps(synthetic_smoke(args.device), indent=2))
    else:
        result = run_study(root, root / args.config, args.device)
        print(json.dumps({k: result[k] for k in ("status", "maximum_auc_gate_achieved", "selected", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
