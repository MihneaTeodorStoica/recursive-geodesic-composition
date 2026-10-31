"""Evaluate trained common probes on prepared long validation leaves only."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from .information_retention import (
    ARMS,
    CORPORA,
    FeatureBundle,
    build_merger,
    evaluate_common,
    evaluate_recursive,
    extract_rows,
    initialized_probe,
    make_hard_panels,
    sha256_file,
    validate_long_rows,
    validation_leaf_index,
    write_csv,
)


def long_bundle(arm, validation, cfg, root, device):
    model = (
        None
        if arm == "pooling"
        else build_merger(root / cfg["frozen"][arm]["checkpoint"], device)
    )
    parts, meta, roots = [], [], {}
    for ci, corpus in enumerate(CORPORA):
        rows = validation[corpus]
        _, packed, item_meta, item_roots, _ = extract_rows(
            rows,
            list(range(len(rows))),
            arm,
            model,
            device,
            None,
            set(),
            ci,
            validation=True,
            batch_size=16,
        )
        parts.append(packed)
        meta.extend(item_meta)
        roots.update(item_roots)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    return FeatureBundle(
        arm,
        torch.empty((0, 3), dtype=torch.int32).numpy(),
        torch.empty(0, 3, 768, dtype=torch.float16),
        torch.empty((0, 2), dtype=torch.int32).numpy(),
        torch.empty(0, 768, dtype=torch.float16),
        torch.cat(parts),
        meta,
        roots,
        None,
        {},
    )


def main():
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(
        (root / "configs/information_retention_validation.yaml").read_text()
    )
    output = root / cfg["output_dir"]
    cache = root / cfg["long_validation"]["cache_dir"]
    if not output.is_dir() or not (output / "RUN_STATUS.json").is_file():
        raise FileNotFoundError("completed short-validation output is required")
    validation, sidecars = {}, {}
    for corpus in CORPORA:
        path = cache / f"{corpus}.pt"
        rows = torch.load(path, map_location="cpu", weights_only=False)
        validate_long_rows(rows)
        sidecar = json.loads(Path(str(path) + ".json").read_text())
        if sidecar["source_start"] != 12000 or sidecar["source_count"] != 1000:
            raise ValueError("long shard is outside the pinned validation range")
        if sidecar["shard_sha256"] != sha256_file(path):
            raise ValueError(f"long shard hash mismatch: {path}")
        validation[corpus] = rows
        sidecars[corpus] = sidecar
        train = torch.load(
            root / cfg["cache_dir"] / corpus / "train.pt",
            map_location="cpu",
            weights_only=False,
        )
        train_ids = {row["id"] for row in train}
        overlap = train_ids & {row["id"] for row in rows}
        if overlap:
            raise AssertionError(
                f"long validation/training document overlap: {sorted(overlap)[:3]}"
            )
        del train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = {arm: long_bundle(arm, validation, cfg, root, device) for arm in ARMS}
    pooling = bundles["pooling"]
    child_target = pooling.validation_triples[:, 1:].reshape(-1, 768)
    child_meta = [
        dict(item, slot=slot) for item in pooling.validation_meta for slot in (0, 1)
    ]
    child_panel = make_hard_panels(
        child_target,
        child_meta,
        int(cfg["retrieval"]["panel_size"]),
        int(cfg["retrieval"]["neighbors_considered"]),
        device,
    )
    _, leaf_meta, leaf_vectors = validation_leaf_index(validation)
    leaf_panel = make_hard_panels(
        leaf_vectors,
        leaf_meta,
        int(cfg["retrieval"]["panel_size"]),
        int(cfg["retrieval"]["neighbors_considered"]),
        device,
    )
    rows, summaries = [], []
    for arm in ARMS:
        probe, _ = initialized_probe("common", 768, cfg, device)
        checkpoint = torch.load(
            output / "probes" / f"common_{arm}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint["updates"] != int(cfg["common_probe"]["updates"]):
            raise ValueError(
                "long evaluation requires the complete fixed-budget common probe"
            )
        probe.load_state_dict(checkpoint["state_dict"], strict=True)
        probe.eval()
        item_rows, item_summary = evaluate_common(
            probe, bundles[arm], child_panel, device, cfg
        )
        for row in item_rows:
            row["task"] = "long_common_child"
        for item in item_summary:
            item["task"] = "long_common_child"
        rows.extend(item_rows)
        summaries.extend(item_summary)
        item_rows, item_summary = evaluate_recursive(
            probe,
            bundles[arm],
            validation,
            leaf_meta,
            leaf_vectors,
            leaf_panel,
            device,
            cfg,
        )
        for row in item_rows:
            row["task"] = "long_recursive_leaf"
        for item in item_summary:
            item["task"] = "long_recursive_leaf"
        rows.extend(item_rows)
        summaries.extend(item_summary)
    write_csv(output / "long_metrics_examples.csv", rows)
    write_csv(output / "long_metrics_summary.csv", summaries)
    (output / "long_metrics.json").write_text(
        json.dumps({"summaries": summaries, "sidecars": sidecars}, indent=2) + "\n"
    )
    run = json.loads((output / "RUN_STATUS.json").read_text())
    run["status"] = "validation_complete"
    run["long_validation"] = {
        "status": "complete_sparse",
        "training_documents_substituted": False,
        "documents": {corpus: len(rows) for corpus, rows in validation.items()},
        "by_bucket": {corpus: sidecars[corpus]["by_bucket"] for corpus in CORPORA},
        "note": "The pinned 1,000-index validation ranges contained fewer than the 50-per-bucket target.",
    }
    (output / "RUN_STATUS.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n"
    )

    def value(arm, task, metric):
        return next(
            x
            for x in summaries
            if x["arm"] == arm
            and x["task"] == task
            and x["metric"] == metric
            and x["slice_type"] == "overall"
        )

    section = [
        "",
        "## Long held-out validation (8k-32k; sparse fixed-range panel)",
        "",
        "Only original <=510-token leaf embeddings were targets; no full-span teacher target was created. "
        "The pinned validation ranges yielded 30 documents, fewer than the aspirational 50 per corpus/bucket.",
        "",
        "| arm | child cosine | child R@1 | recursive leaf cosine | recursive leaf R@1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        section.append(
            f"| {arm} | {value(arm, 'long_common_child', 'cosine')['value']:.4f} | "
            f"{value(arm, 'long_common_child', 'recall_at_1')['value']:.4f} | "
            f"{value(arm, 'long_recursive_leaf', 'cosine')['value']:.4f} | "
            f"{value(arm, 'long_recursive_leaf', 'recall_at_1')['value']:.4f} |"
        )
    with (output / "REPORT.md").open("a") as handle:
        handle.write("\n".join(section) + "\n")
    files = sorted(
        p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(output)}\n" for p in files)
    )
    print(json.dumps(run["long_validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
