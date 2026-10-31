"""Independent, document-disjoint discriminator evaluation."""
from __future__ import annotations

import argparse
import json
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

from .architecture import GeodesicMerge


# Frozen before future test access.  The RBF gamma is 1 / 768, the standard
# inverse-feature-count default for this experiment's standardized embeddings.
DEFAULT_ATTACK_CONFIG = {
    "protocol_version": 1,
    "split_seed": 1729,
    "model_seed": 1729,
    "test_size": 0.5,
    "logistic_c": 1.0,
    "logistic_max_iter": 1000,
    "rbf_c": 1.0,
    "rbf_gamma": 1.0 / 768.0,
}


def roots(model, rows, device):
    out = []
    teachers = []
    ids = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            if not torch.isfinite(r["leaves"]).all() or not torch.isfinite(r["lengths"]).all():
                raise FloatingPointError(f"non-finite cached test leaves/lengths at row {i} id={r.get('id')}")
            x = r["leaves"].to(device)
            lengths = r["lengths"].to(device)
            nodes = x[None]
            lens = lengths[None]
            while nodes.shape[1] > 1:
                pair_count = nodes.shape[1] // 2
                old_nodes, old_lens = nodes, lens
                parents = model.merge(
                    old_nodes[:, : pair_count * 2].reshape(-1, 2, model.d),
                    old_lens[:, : pair_count * 2].reshape(-1, 2),
                )
                if not torch.isfinite(parents).all():
                    raise FloatingPointError(f"non-finite model root during evaluation at row {i} id={r.get('id')}")
                nodes = parents.reshape(1, pair_count, model.d)
                lens = old_lens[:, : pair_count * 2].reshape(-1, 2).sum(-1).reshape(1, pair_count)
                if old_nodes.shape[1] % 2:
                    nodes = torch.cat([nodes, old_nodes[:, -1:, :]], dim=1)
                    lens = torch.cat([lens, old_lens[:, -1:]], dim=1)
            root_target = next((target for target in reversed(r["targets"]) if target is not None), None)
            if root_target is None:
                continue
            if not torch.isfinite(root_target).all():
                raise FloatingPointError(f"non-finite cached test target at row {i} id={r.get('id')}")
            out.append(nodes[0, 0].cpu().numpy())
            teachers.append(root_target.numpy())
            ids.append(r["id"])
    return np.asarray(out), np.asarray(teachers), np.asarray(ids)


def bootstrap_auc(y, scores, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    m = len(y)
    for _ in range(n):
        indexes = rng.integers(0, m, m)
        if len(np.unique(y[indexes])) == 2:
            vals.append(roc_auc_score(y[indexes], scores[indexes]))
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [None, None]


def document_group_split(real, fake, groups, *, seed, test_size):
    """Build paired labels and split whole documents, never individual vectors."""
    real = np.asarray(real)
    fake = np.asarray(fake)
    groups = np.asarray(groups)
    if real.shape != fake.shape or len(real) != len(groups):
        raise ValueError("real, fake, and document groups must describe matched rows")
    if len(real) < 4 or not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise FloatingPointError(f"invalid evaluation arrays: examples={len(real)}")
    x = np.vstack([real, fake])
    y = np.r_[np.ones(len(real), dtype=int), np.zeros(len(fake), dtype=int)]
    paired_groups = np.r_[groups, groups]
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(x, y, paired_groups)
    )
    if set(paired_groups[train]).intersection(paired_groups[test]):
        raise AssertionError("document-group leakage in discriminator split")
    return x, y, paired_groups, train, test


def _attack_metrics(model, x, y, train, test, bootstrap, bootstrap_seed):
    model.fit(x[train], y[train])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x[test])[:, 1]
    else:
        scores = model.decision_function(x[test])
    predictions = model.predict(x[test])
    auc = float(roc_auc_score(y[test], scores))
    return {
        "auc": auc,
        "auc_ci95": bootstrap_auc(y[test], scores, bootstrap, bootstrap_seed) if bootstrap else None,
        "pr_auc": float(average_precision_score(y[test], scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], predictions)),
    }


def evaluate_attacks(real, fake, groups, *, config=None, bootstrap=0):
    """Run the two predeclared attacks and select maximum held-out ROC-AUC."""
    attack_config = {**DEFAULT_ATTACK_CONFIG, **(config or {})}
    x, y, paired_groups, train, test = document_group_split(
        real,
        fake,
        groups,
        seed=int(attack_config["split_seed"]),
        test_size=float(attack_config["test_size"]),
    )
    model_seed = int(attack_config["model_seed"])
    models = {
        "standard_scaled_logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=float(attack_config["logistic_c"]),
                max_iter=int(attack_config["logistic_max_iter"]),
                solver="lbfgs",
                random_state=model_seed,
            )),
        ]),
        "standard_scaled_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=float(attack_config["rbf_c"]),
                gamma=float(attack_config["rbf_gamma"]),
                kernel="rbf",
                random_state=model_seed,
            )),
        ]),
    }
    attacks = {
        name: _attack_metrics(model, x, y, train, test, bootstrap, model_seed)
        for name, model in models.items()
    }
    strongest = max(attacks, key=lambda name: attacks[name]["auc"])
    return {
        "attack_protocol": attack_config,
        "attacks": attacks,
        "strongest_attack": strongest,
        "max_attack_auc": attacks[strongest]["auc"],
        "disc_train_examples": int(len(train)),
        "disc_test_examples": int(len(test)),
        "disc_train_documents": int(len(np.unique(paired_groups[train]))),
        "disc_test_documents": int(len(np.unique(paired_groups[test]))),
    }


def _load_attack_config(path):
    if path is None:
        return None
    loaded = yaml.safe_load(Path(path).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("attack config must be a YAML mapping")
    unknown = set(loaded).difference(DEFAULT_ATTACK_CONFIG)
    if unknown:
        raise ValueError(f"unknown attack config keys: {sorted(unknown)}")
    return loaded


def evaluate(path, device, attack_config_path=None):
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    rows = torch.load(Path(cfg["cache_dir"]) / "test.pt", weights_only=False)
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = GeodesicMerge(
        d=rows[0]["leaves"].shape[-1],
        hidden=cfg["encoder_hidden"],
        layers=cfg["encoder_layers"],
        heads=cfg["heads"],
        max_angle=float(cfg.get("max_geodesic_angle", cfg.get("max_angle", 0.8))),
        decoder=ck["variant"] == "autoencoder",
        decoder_layers=cfg["decoder_layers"],
        decoder_length_mode=str(cfg.get("decoder_length_mode", "oracle")),
        ratio_log_bound=float(cfg.get("ratio_log_bound", 4.0)),
        ratio_loss_weight=float(cfg.get("ratio_loss_weight", 0.0)),
        relational_loss_weight=float(cfg.get("relational_loss_weight", 0.0)),
        importance_mode=str(cfg.get("importance_mode", "token")),
        importance_beta=float(cfg.get("importance_beta", 1.0)),
        decoder_mode=str(cfg.get("decoder_mode", "legacy")),
        importance_loss_weight=float(cfg.get("importance_loss_weight", 0.0)),
        importance_prior_weight=float(cfg.get("importance_prior_weight", 0.0)),
        delta_loss_weight=float(cfg.get("delta_loss_weight", 0.0)),
        sharpness_loss_weight=float(cfg.get("sharpness_loss_weight", 0.0)),
        slot_loss_weight=float(cfg.get("slot_loss_weight", 0.0)),
        sharpness_margin=float(cfg.get("sharpness_margin", 0.1)),
        geodesic_parameterization=str(cfg.get("geodesic_parameterization", "legacy")),
    ).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    fake, real, groups = roots(model, rows, dev)
    boot = int(cfg.get("bootstrap", 2000))
    attack_result = evaluate_attacks(
        real,
        fake,
        groups,
        config=_load_attack_config(attack_config_path),
        bootstrap=boot,
    )
    primary = attack_result["attacks"][attack_result["strongest_attack"]]
    result = {
        "variant": ck["variant"],
        "seed": ck.get("seed"),
        "documents": len(rows),
        **attack_result,
        # Compatibility keys now intentionally represent the predeclared
        # strongest attack, which is the primary detectability estimate.
        "auc": primary["auc"],
        "auc_ci95": primary["auc_ci95"],
        "pr_auc": primary["pr_auc"],
        "balanced_accuracy": primary["balanced_accuracy"],
        "teacher_cosine": float(np.mean(np.sum(fake * real, axis=1) / (np.linalg.norm(fake, axis=1) * np.linalg.norm(real, axis=1)))),
        "fake_norm_mean": float(np.linalg.norm(fake, axis=1).mean()),
        "real_norm_mean": float(np.linalg.norm(real, axis=1).mean()),
    }
    out = Path(cfg["output_dir"]) / f"{ck['variant']}_seed{ck.get('seed')}_evaluation.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attack-config", default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.device, args.attack_config)
