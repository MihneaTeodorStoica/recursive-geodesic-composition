#!/usr/bin/env python3
"""Generate the two external final-paper figures from committed result artifacts.

Figure 1 is generated directly in the paper's LaTeX/TikZ source. This script
produces external Figures 2 and 3 only. It is deterministic for a fixed
Python/package stack and artifact tree, and it contains no result-value
fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# Okabe-Ito colorblind-safe palette, plus neutral grays.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
BLACK = "#222222"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"

DPI = 300
AUTHOR = "Anonymous authors"
CREATOR = "recursive-geodesic-composition/figures/generate_figures.py"
EXPECTED_STEMS = (
    "fig2_official_per_corpus",
    "fig3_kary_scaling",
)

SOURCES = {
    "official_v6": PROJECT / "results/official_v6/results.json",
    "kary_scaling": PROJECT / "results/kary_scaling/KARY_VALIDATION_REPORT.json",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": BLACK,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "axes.axisbelow": True,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "text.color": BLACK,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.75,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
        "svg.fonttype": "none",
        "svg.hashsalt": "recursive-geodesic-composition-v1-paper",
        "path.simplify": False,
    }
)


def load_json(key: str) -> dict[str, Any]:
    path = SOURCES[key]
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def finite(values: Iterable[float], context: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"Missing/non-finite values for {context}: {result}")
    return result


def require_role(data: dict[str, Any], key: str, expected: Any, context: str) -> None:
    """Check a role assertion carried by an artifact."""
    if data.get(key) != expected:
        raise ValueError(f"{context} must declare {key}={expected!r}")


def style_axis(ax: mpl.axes.Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y")
    ax.tick_params(length=3, width=0.7)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_values(
    ax: mpl.axes.Axes,
    xs: Iterable[float],
    ys: Iterable[float],
    *,
    fmt: str = ".3f",
    dy: float = 0.0,
    fontsize: float = 7.8,
) -> None:
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y + dy, format(y, fmt), ha="center", va="bottom", fontsize=fontsize)


def _make_svg_accessible(path: Path, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find("<svg")
    end = text.find(">", start)
    if start < 0 or end < 0:
        raise ValueError(f"Could not find SVG root in {path}")
    stem = path.stem.replace("_", "-")
    root = text[start:end]
    root += f' role="img" aria-labelledby="{stem}-title {stem}-desc"'
    accessibility = (
        f">\n<title id=\"{stem}-title\">{escape(title)}</title>\n"
        f"<desc id=\"{stem}-desc\">{escape(description)}</desc>"
    )
    accessible = text[:start] + root + accessibility + text[end + 1 :]
    path.write_text(
        "\n".join(line.rstrip() for line in accessible.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_figure(fig: mpl.figure.Figure, stem: str, title: str, description: str) -> None:
    png = HERE / f"{stem}.png"
    svg = HERE / f"{stem}.svg"
    common = {"facecolor": "white", "edgecolor": "none"}
    fig.savefig(
        png,
        dpi=DPI,
        metadata={
            "Title": title,
            "Description": description,
            "Author": AUTHOR,
            "Creator": CREATOR,
            "Software": CREATOR,
        },
        **common,
    )
    fig.savefig(
        svg,
        metadata={"Date": None, "Creator": CREATOR},
        **common,
    )
    plt.close(fig)
    _make_svg_accessible(svg, title, description)


def figure_official_per_corpus(v6: dict[str, Any]) -> None:
    require_role(v6, "official_test_evaluated", True, "Official v6 per-corpus source")
    require_role(v6, "test_role_only", True, "Official v6 per-corpus source")
    corpora = ["fineweb_edu", "fineweb", "openwebtext", "wikipedia_en_20231101"]
    labels = ["FineWeb-Edu", "FineWeb", "OpenWebText", "Wikipedia"]
    aggregates = v6["three_seed_scope_aggregates"]
    means = finite(
        (aggregates[corpus]["mean_auc"] for corpus in corpora),
        "v6 per-corpus mean AUC",
    )
    cis = [
        finite(aggregates[corpus]["ci95"], f"v6 {corpus} AUC CI")
        for corpus in corpora
    ]
    for mean, ci in zip(means, cis, strict=True):
        if not ci[0] <= mean <= ci[1]:
            raise ValueError("Official v6 per-corpus CI does not contain its estimate")
    seed_values = [
        finite(
            (
                seed["safe_arm"]["scopes"][corpus]["attacks"]
                ["strongest_orientation_safe_auc"]
                for seed in v6["seeds"]
            ),
            f"v6 {corpus} seed AUC",
        )
        for corpus in corpora
    ]

    x = np.arange(len(corpora), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.18)
    for i, (mean, ci, seeds) in enumerate(
        zip(means, cis, seed_values, strict=True)
    ):
        ax.errorbar(
            x[i],
            mean,
            yerr=np.array([[mean - ci[0]], [ci[1] - mean]]),
            fmt="o",
            ms=8,
            color=GREEN,
            ecolor=GREEN,
            elinewidth=2,
            capsize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        for offset, value in zip((-0.07, 0.0, 0.07), seeds, strict=True):
            ax.scatter(
                x[i] + offset,
                value,
                s=22,
                facecolor="white",
                edgecolor=GREEN,
                linewidth=0.9,
                zorder=2,
            )
        ax.text(
            x[i],
            ci[1] + 0.006,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    ax.axhline(
        0.50,
        color=MID_GRAY,
        linestyle=(0, (4, 3)),
        linewidth=1,
        label="Chance",
    )
    ax.axhline(
        0.60,
        color=BLUE,
        linestyle=(0, (5, 3)),
        linewidth=1,
        label="Sole AUC target",
    )
    ax.set_ylim(0.495, 0.675)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Strongest orientation-safe ROC-AUC ↓")
    ax.set_title("Official v6 safe-arm per-corpus detectability")
    ax.legend(loc="upper left")
    ax.text(
        0.99,
        0.96,
        "○ seeds   ● mean with shared-group 95% CI",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MID_GRAY,
    )
    style_axis(ax)

    save_figure(
        fig,
        EXPECTED_STEMS[0],
        "Official v6 safe-arm strongest AUC by corpus",
        "Official v6 strongest orientation-safe fixed-attack ROC-AUC for four "
        "corpora. Open symbols are individual checkpoints; filled symbols are "
        "three-seed means with 2000-replicate shared-held-out-document-group "
        "bootstrap 95 percent confidence intervals. The only AUC target shown "
        "is below 0.60. Wikipedia remains the most distinguishable corpus.",
    )


def figure_kary_scaling(kary: dict[str, Any]) -> None:
    require_role(kary, "validation_only", True, "K-ary source")
    require_role(kary, "official_test_evaluated", False, "K-ary source")
    keys = ["binary_reference", "kary4_reference", "kary8"]
    labels = ["K=2", "K=4", "K=8"]
    colors = [SKY, BLUE, ORANGE]
    markers = ["o", "s", "D"]
    models = [kary["models"][key] for key in keys]
    x = np.arange(3)

    auc = finite(
        (model["orientation_safe_strongest_attack_auc"] for model in models),
        "K-ary AUC",
    )
    cosine = finite(
        (model["macro_teacher_cosine"] for model in models), "K-ary cosine"
    )
    depth_mean = finite(
        (model["tree_depth_mean"] for model in models), "K-ary mean depth"
    )
    depth_max = finite(
        (model["tree_depth_max"] for model in models), "K-ary max depth"
    )
    throughput = finite(
        (
            model["validation_root_throughput"]["documents_per_second"]
            for model in models
        ),
        "K-ary throughput",
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), constrained_layout=True)
    ax = axes[0, 0]
    for i, value in enumerate(auc):
        ax.scatter(
            x[i],
            value,
            s=72,
            color=colors[i],
            marker=markers[i],
            edgecolor=BLACK,
            linewidth=0.6,
            zorder=3,
        )
    ax.axhline(0.50, color=MID_GRAY, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_ylim(0.495, 0.70)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Strongest ROC-AUC ↓")
    ax.set_title("Detectability (point estimates)")
    annotate_values(ax, x, auc, fmt=".3f", dy=0.008)
    style_axis(ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    for i, value in enumerate(cosine):
        ax.scatter(
            x[i],
            value,
            s=72,
            color=colors[i],
            marker=markers[i],
            edgecolor=BLACK,
            linewidth=0.6,
            zorder=3,
        )
    ax.axhline(0.975, color="#D55E00", linestyle=(0, (2, 2)), linewidth=1)
    ax.set_ylim(0.970, 0.983)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro teacher cosine ↑")
    ax.set_title("Semantic fidelity (point estimates)")
    annotate_values(ax, x, cosine, fmt=".4f", dy=0.00035)
    style_axis(ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    ax.bar(
        x,
        depth_mean,
        color=colors,
        width=0.64,
        edgecolor=BLACK,
        linewidth=0.5,
        label="Mean",
    )
    ax.scatter(
        x,
        depth_max,
        marker="_",
        s=260,
        linewidth=2.2,
        color=BLACK,
        label="Maximum",
        zorder=3,
    )
    ax.set_ylim(0, 4.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Tree depth ↓")
    ax.set_title("Recursive depth")
    annotate_values(ax, x, depth_mean, fmt=".2f", dy=0.11)
    ax.legend(loc="upper right")
    style_axis(ax)
    panel_label(ax, "C")

    ax = axes[1, 1]
    ax.bar(x, throughput, color=colors, width=0.64, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylim(0, 5900)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation roots / second ↑")
    ax.set_title("Matched-cache GPU throughput")
    annotate_values(ax, x, throughput, fmt=".0f", dy=130)
    style_axis(ax)
    panel_label(ax, "D")

    save_figure(
        fig,
        EXPECTED_STEMS[1],
        "Validation-only K-ary scaling",
        "Matched-row comparison of binary, four-way, and eight-way recursive "
        "mergers with 768-token leaves. K=4 improves depth and throughput over "
        "K=2 on this cache, while K=8 is more detectable and was rejected by "
        "the training saturation guard.",
    )


def validate_outputs() -> None:
    failures: list[str] = []
    for stem in EXPECTED_STEMS:
        png = HERE / f"{stem}.png"
        svg = HERE / f"{stem}.svg"
        if not png.is_file() or png.stat().st_size < 20_000:
            failures.append(f"missing or unexpectedly small PNG: {png.name}")
        else:
            with Image.open(png) as image:
                if image.format != "PNG" or image.width < 1800 or image.height < 900:
                    failures.append(f"invalid/low-resolution PNG: {png.name} {image.size}")
                if not image.info.get("Title") or not image.info.get("Description"):
                    failures.append(f"missing PNG accessibility metadata: {png.name}")
                if image.info.get("Author") != AUTHOR:
                    failures.append(f"incorrect PNG author metadata: {png.name}")
                if image.info.get("Creator") != CREATOR:
                    failures.append(f"incorrect PNG creator metadata: {png.name}")
        if not svg.is_file() or svg.stat().st_size < 15_000:
            failures.append(f"missing or unexpectedly small SVG: {svg.name}")
        else:
            text = svg.read_text(encoding="utf-8")
            if (
                "<svg" not in text
                or 'role="img"' not in text
                or "<title id=" not in text
                or "<desc id=" not in text
            ):
                failures.append(f"missing SVG accessibility markup: {svg.name}")
            if "<image" in text:
                failures.append(f"SVG unexpectedly embeds raster data: {svg.name}")
            if CREATOR not in text:
                failures.append(f"incorrect SVG creator metadata: {svg.name}")
    if failures:
        raise RuntimeError("Output validation failed:\n- " + "\n- ".join(failures))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for key, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required machine-readable source is missing ({key}): {path}"
            )

    figure_official_per_corpus(load_json("official_v6"))
    figure_kary_scaling(load_json("kary_scaling"))
    validate_outputs()

    print("Generated and validated external final-paper figures:")
    for stem in EXPECTED_STEMS:
        for suffix in ("svg", "png"):
            path = HERE / f"{stem}.{suffix}"
            print(
                f"  {path.name:42s} {path.stat().st_size:8d} bytes  "
                f"sha256={sha256(path)[:16]}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
