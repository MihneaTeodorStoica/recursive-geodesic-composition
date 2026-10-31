#!/usr/bin/env python3
"""Render the fixed source-versus-final comparison; never fit or select models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_continuation_comparison import CONFIG, load_protocol
from safe_baselines_protocol import ROOT, atomic_json, require_hash, sha256


def interval(cell):
    return f"{cell['mean']:+.9f} [{cell['ci95'][0]:+.9f}, {cell['ci95'][1]:+.9f}]"


def render(path):
    cfg, _, _, _, _ = load_protocol(path)
    root = ROOT / cfg["output_dir"]
    r = json.loads((root / "RESULTS.json").read_text())
    require_hash(root / "POINTS.json", r["points_sha256"])
    points = json.loads((root / "POINTS.json").read_text())
    t = r["teacher"]["three_seed_mean"]["pooled"]
    p = r["reconstruction"]["three_seed_mean"]["pooled"]
    a = r["source_detection"]["pooled"]["three_seed_mean"]
    cells = [("Teacher cosine (4,000)", t["source"], t["final"], t["delta"]),
             ("Source AUC (2,000 held out)", a["source"], a["final"], a["final_minus_source"]),
             ("Parent-copy cosine (606)", p["source_copy"], p["final_copy"], p["copy_delta"]),
             ("Probe cosine (606)", p["source_probe"], p["final_probe"], p["probe_delta"]),
             ("Probe gain over copy (606)", p["source_gain"], p["final_gain"], p["gain_delta"]),
             ("Probe order gap (606)", p["source_order_gap"], p["final_order_gap"], p["order_gap_delta"])]
    lines = ["# Matched source-versus-final continuation comparison", "",
             "The released initialization-source merger versus its fixed step-400 clean-v6 final checkpoint, across all three lineage-matched pairs. This evaluates the continuation bundle as a whole, not a decoder-objective ablation or training from scratch.", "",
             "| Metric (documents) | Source | Final | Final minus source [paired 95% CI] |", "| --- | ---: | ---: | --- |"]
    tex = [r"% Generated from fixed RESULTS.json; document-paired intervals.", r"\begin{table}[H]", r"\caption{Matched continuation comparison on regenerated teacher-comparable prefixes. Teacher cosine uses all 4,000 documents; pooled source AUC uses the same 2,000 held-out document pairs at each stage. Reconstruction uses the same 606 exact-four prefixes and each stage's separately fitted, released TRAIN probe. Gain subtracts copying that stage's own parent; order gap subtracts the mean over all 24 target assignments. Lower AUC is preferred; higher cosine/gain is preferred. Intervals condition on the fixed checkpoints, probes and selected classifiers.}", r"\label{tab:continuation-comparison}", r"\centering\footnotesize",
           r"\begin{tabularx}{\linewidth}{Yrrr}", r"\toprule", r"Metric (documents) & Source & Final & Final $-$ source [95\% CI] \\", r"\midrule"]
    for label, src, fin, delta in cells:
        lines.append(f"| {label} | {src['mean']:.9f} | {fin['mean']:.9f} | {interval(delta)} |")
        tex.append(f"{label} & {src['mean']:.6f} & {fin['mean']:.6f} & {delta['mean']:+.6f} [{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}] " + r"\\")
    tex += [r"\bottomrule", r"\end{tabularx}",
            r"\end{table}"]
    (root / "continuation_comparison_table.tex").write_text("\n".join(tex) + "\n")
    lines += ["", "## Parent-vector diagnostics", "",
              f"- Source/final parent agreement: {interval(t['output_agreement'])}.",
              f"- Source minus weighted pooling: {interval(t['source_minus_weighted'])}.",
              f"- Final minus weighted pooling: {interval(t['final_minus_weighted'])}.",
              f"- Fraction of documents with positive checkpoint-mean teacher change: {t['delta']['fraction_positive']:.9f}.",
              f"- Fraction of exact-four documents with positive checkpoint-mean probe gain change: {p['gain_delta']['fraction_positive']:.9f}.",
              "", "## Teacher fidelity: all checkpoints and prespecified strata", "",
              "| Checkpoint | Scope | Documents | Source | Final | Change [95% CI] | Fraction positive |", "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for checkpoint, scopes in r["teacher"].items():
        for scope in r["fidelity_scope_order"]:
            c = scopes[scope]
            lines.append(f"| {checkpoint} | {scope} | {c['delta']['documents']} | {c['source']['mean']:.9f} | {c['final']['mean']:.9f} | {interval(c['delta'])} | {c['delta']['fraction_positive']:.6f} |")
    lines += ["", "## Source detectability: fixed logistic/RBF classifier recipe", "",
              "Two fresh source classifiers per checkpoint/scope (30 fits); all final classifiers reused with file-hash checks from improvement 1. Identical document-grouped 50:50 splits. Strongest observed family is fixed within resampling, independently for each stage/checkpoint.", "",
              "| Checkpoint | Scope | Held-out documents | Source AUC | Final AUC | Change [95% CI] | Source / final selected family |", "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for scope, c in r["source_detection"].items():
        for seed, row in zip(cfg["seeds"], c["per_checkpoint"]):
            families = c["selected_families"][str(seed)]
            lines.append(f"| {seed} | {scope} | {c['holdout_documents']} | {row['source']['mean']:.9f} | {row['final']['mean']:.9f} | {interval(row['final_minus_source'])} | {families['source']} / {families['final']} |")
        row = c["three_seed_mean"]
        lines.append(f"| three_seed_mean | {scope} | {c['holdout_documents']} | {row['source']['mean']:.9f} | {row['final']['mean']:.9f} | {interval(row['final_minus_source'])} | fixed per checkpoint |")
    lines += ["", "## Reconstruction: all checkpoints and corpora", "",
              "| Checkpoint | Scope | Source probe | Final probe | Probe change [95% CI] | Source gain over copy | Final gain over copy | Gain change [95% CI] | Order-gap change [95% CI] |", "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |"]
    for checkpoint, scopes in r["reconstruction"].items():
        for scope in r["scope_order"]:
            c = scopes[scope]
            lines.append(f"| {checkpoint} | {scope} | {c['source_probe']['mean']:.9f} | {c['final_probe']['mean']:.9f} | {interval(c['probe_delta'])} | {c['source_gain']['mean']:.9f} | {c['final_gain']['mean']:.9f} | {interval(c['gain_delta'])} | {interval(c['order_gap_delta'])} |")
    lines += ["", "## Scope and uncertainty", "",
              "- Paired endpoint comparison: pretrained source merger, discarded source decoder, freshly initialized decoder, 300 decoder-only updates, then 100 joint updates with encoder LR scale 0.02. It cannot isolate the decoder loss, distribution penalties, schedule, or other training components.",
              "- Released source/candidate probes share documented architecture, initialization and TRAIN fitting recipe. Their TRAIN selection and minibatch schedule metadata are checked; no probes are fitted here. This is not a comparison with the distinct historical matched-pooling probes.",
              "- 2,000 shared document bootstrap resamples, paired across stages and checkpoints. Scalar checkpoint scores are averaged within documents; AUCs are computed per checkpoint and averaged, never from averaged classifier scores. Fixed classifier family choices; no classifier refitting or family reselection in bootstrap.",
              "- AUC bootstrap counts weight both observations of each teacher/generated document pair. Orientation-safe AUC is recomputed in each resample under the original convention. This is conditional uncertainty, not a simultaneous multiple-comparison guarantee or new-training-seed uncertainty.",
              "- All methods use the same regenerated inputs. Final parent/probe tensors and final classifier scores are reused unchanged. No original embedding-byte identity is asserted. These already evaluated documents are not a fresh confirmatory test.",
              "- Child cosine, gain over copy, and the order gap are local diagnostics, not downstream utility, losslessness, or document-specific order storage. No threshold crossing or favorable checkpoint is used for model selection.",
              "- No merger, decoder, or probe updates; no teacher encoding. Both source classifier families, all checkpoints, corpora, and arities are retained regardless of sign."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), sharey=True, constrained_layout=True)
    for ax, values, label in ((axes[0], points["teacher"]["three_seed_mean"]["delta"], "Teacher cosine: final minus source\n(4,000 documents)"),
                              (axes[1], points["reconstruction"]["three_seed_mean"]["gain_delta"], "Probe gain over copy: final minus source\n(606 exact-four documents)")):
        values = np.sort(values)
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", linewidth=1.3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
        ax.set(xlabel=label, ylim=(0, 1.01))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Fraction of documents")
    fig.savefig(root / "continuation_ecdf.png", dpi=180)
    plt.close(fig)
    atomic_json(root / "ARTIFACT_HASHES.json", {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "ARTIFACT_HASHES.json"})
    print(f"Rendered {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    render(parser.parse_args().config)
