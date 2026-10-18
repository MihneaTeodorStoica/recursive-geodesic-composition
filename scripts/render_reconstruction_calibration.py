#!/usr/bin/env python3
"""Render saved reconstruction calibration; no inference, selection or fitting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_reconstruction_calibration import CONFIG, LABELS, load_protocol
from safe_baselines_protocol import ROOT, atomic_json, require_hash, sha256


def render(config_path):
    cfg, _, _, _, _ = load_protocol(config_path)
    root = ROOT / cfg["output_dir"]
    result = json.loads((root / "RESULTS.json").read_text())
    require_hash(root / "DOCUMENT_SCORES.json", result["document_scores_sha256"])
    docs = json.loads((root / "DOCUMENT_SCORES.json").read_text())
    pooled = result["summaries"]["three_seed_mean"]["pooled"]
    methods = pooled["methods"]
    lines = ["# Parent-copy and target-slot permutation calibration", "",
             "606 identity-matched exact-four safe prefixes; all three frozen final checkpoints, their built-in decoders, and the released candidate TRAIN-fitted probes. No new training or teacher encoding. Parents and children are reused from the preceding matched analyses.", "",
             "| Prediction rule | Ordered child cosine | Difference vs learned-parent copy | All-24 shuffled cosine | Ordered minus shuffled |",
             "| --- | ---: | ---: | ---: | ---: |"]
    tex = [r"% Generated from RESULTS.json; checkpoint scores averaged within documents.", r"\begin{table}[H]", r"\caption{Matched reconstruction calibration on 606 four-child prefixes. Learned rows average checkpoint scores, not vectors. Copy rows repeat one unit parent in all four output slots. $\Delta$ compares each row with learned-parent copy; only the decoder/probe rows isolate a decoding gain for that same parent. Shuffled scoring averages all 24 within-document target permutations, holding inputs and predictions fixed. Order gap is ordered minus shuffled cosine.}", r"\label{tab:reconstruction-calibration}", r"\centering\small",
           r"\begin{tabularx}{\linewidth}{Yrrrr}", r"\toprule",
           r"Prediction rule & Child cosine & $\Delta$ vs copy & Shuffled & Order gap \\", r"\midrule"]
    for method in cfg["methods"]:
        m = methods[method]
        ordered, shuffled, gap = (m[key]["mean"] for key in ("ordered", "shuffled_all24", "ordered_minus_shuffled"))
        gain = ordered - methods["parent_copy"]["ordered"]["mean"]
        lines.append(f"| {LABELS[method]} | {ordered:.9f} | {gain:+.9f} | {shuffled:.9f} | {gap:+.9f} |")
        tex.append(f"{LABELS[method]} & {ordered:.6f} & {gain:+.6f} & {shuffled:.6f} & {gap:+.8f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabularx}",
            r"\end{table}"]
    (root / "reconstruction_calibration_table.tex").write_text("\n".join(tex) + "\n")
    lines += ["", "## Paired reconstruction contrasts", "", "| Contrast | Mean | Conditional paired 95% CI | Fraction positive |", "| --- | ---: | --- | ---: |"]
    for name, cell in pooled["contrasts"].items():
        lines.append(f"| {name} | {cell['mean']:+.9f} | [{cell['ci95'][0]:+.9f}, {cell['ci95'][1]:+.9f}] | {cell['fraction_positive']:.6f} |")
    lines += ["", "## Exact order controls", "",
              "All 24 target-slot permutations include the identity; each target lands in any slot with probability 1/4. The nine derangements exclude every original slot match. They are scored after inference, never fed to a decoder and never treated as independent documents.", "",
              "| Method | Ordered minus all-24 [95% CI] | Ordered minus deranged-9 [95% CI] | Oracle best assignment | Oracle best minus ordered |", "| --- | --- | --- | ---: | ---: |"]
    for method, m in methods.items():
        cells = [m[key] for key in ("ordered_minus_shuffled", "ordered_minus_deranged")]
        intervals = [f"{c['mean']:+.9f} [{c['ci95'][0]:+.9f}, {c['ci95'][1]:+.9f}]" for c in cells]
        lines.append(f"| {LABELS[method]} | {' | '.join(intervals)} | {m['oracle_best_assignment']['mean']:.9f} | {m['oracle_best_minus_ordered']['mean']:.9f} |")
    lines += ["", "## Per-checkpoint and per-corpus summaries", "",
              "| Checkpoint | Scope | Method | Ordered | Gain over same-parent copy [95% CI] | Order gap [95% CI] |", "| --- | --- | --- | ---: | --- | --- |"]
    for seed, scopes in result["summaries"].items():
        for scope in result["scope_order"]:
            s = scopes[scope]
            for method, contrast in (("strict_decoder", "strict_minus_parent_copy"), ("frozen_probe", "probe_minus_parent_copy")):
                c, gap = s["contrasts"][contrast], s["methods"][method]["ordered_minus_shuffled"]
                lines.append(f"| {seed} | {scope} | {method} | {s['methods'][method]['ordered']['mean']:.9f} | {c['mean']:+.9f} [{c['ci95'][0]:+.9f}, {c['ci95'][1]:+.9f}] | {gap['mean']:+.9f} [{gap['ci95'][0]:+.9f}, {gap['ci95'][1]:+.9f}] |")
    geom = result["geometry"]["three_seed_mean"]["pooled"]
    lines += ["", "## Residual-geometry calibration", "",
              f"Both decoder families add a residual with norm at most 0.25 before normalization, giving angular radius asin(0.25) = {np.degrees(np.arcsin(.25)):.6f} degrees.",
              f"- Mean fraction of children outside that radius around their learned parent: {geom['fraction_children_outside_cap']['mean']:.9f}.",
              f"- Mean loose oracle spherical-cap cosine ceiling: {geom['loose_cap_cosine_ceiling']['mean']:.9f}.",
              "This is an upper bound that ignores the tighter coordinate box and available information, not an achievable decoder result or evidence that the bound caused the observed performance.", "",
              "## Historical numerical check", "", "| Seed | Scope | Method | New minus original mean | Within fixed 1e-5 diagnostic tolerance |", "| --- | --- | --- | ---: | --- |"]
    for seed, scopes in result["historical_reproduction_diagnostics"].items():
        for scope, methods_here in scopes.items():
            for method, c in methods_here.items():
                lines.append(f"| {seed} | {scope} | {method} | {c['difference']:+.12f} | {c['within_fixed_diagnostic_tolerance']} |")
    lines += ["", "## Interpretation limits", "",
              "- Copy baselines use no fitted decoder. Unweighted pooling maximizes the unweighted average child cosine among single unit vectors copied into every slot; it does not upper-bound distinct slot-specific predictions.",
              "- All reported contrasts use regenerated, matched inputs. Original official results and the historical seed-424242 matched-pooling probe study remain unchanged and separate.",
              "- The released candidate TRAIN probes are not assumed to be identical to the separately fitted matched-pooling probes. The latter three binaries are absent from the release, so no new pooling-probe score or calibrated gain is claimed for them.",
              "- A positive order gap can reflect corpus-wide positional priors; it does not prove recovery of document-specific child order, lossless storage, or preserved mutual information.",
              "- Oracle best assignment uses the target labels to choose a permutation. It is a diagnostic upper envelope, not a deployed decoder or retrieval accuracy.",
              "- These are local four-child reconstructions from direct parents, not decoding all original leaves from unrestricted recursive roots.",
              "- Intervals use 2,000 shared document resamples, conditional on fixed checkpoints/probes. Checkpoint scores are averaged within each document first. There are 606 document units, not 2,424 child units or 24 permutation replicas per document."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True, constrained_layout=True)
    for method, contrast, label in (("strict_decoder", "strict_minus_parent_copy", "Built-in decoder"), ("frozen_probe", "probe_minus_parent_copy", "Frozen probe")):
        values = [np.array([row["contrasts"]["three_seed_mean"][contrast] for row in docs]),
                  np.array([row["scores"]["three_seed_mean"][method]["ordered_minus_shuffled"] for row in docs])]
        for ax, value in zip(axes, values):
            ax.step(np.sort(value), np.arange(1, len(value) + 1) / len(value), where="post", label=label, linewidth=1.3)
    for ax, label in zip(axes, ("Ordered cosine gain\nover same-parent copy", "Order gap: ordered cosine\nminus all-permutation mean")):
        ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
        ax.set(xlabel=label, ylim=(0, 1.01))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Fraction of documents")
    fig.savefig(root / "calibration_ecdf.png", dpi=180)
    plt.close(fig)
    atomic_json(root / "ARTIFACT_HASHES.json", {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "ARTIFACT_HASHES.json"})
    print(f"Rendered {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    render(parser.parse_args().config)
