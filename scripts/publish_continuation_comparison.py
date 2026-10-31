#!/usr/bin/env python3
"""Layout-only polish of the frozen renderer, without changing analysis bytes.

Post-result presentation adjustments: avoid exponent/axis-label overlap, and
express the table's small changes in 1e-6 units instead of rounding a teacher
interval to two identical six-decimal endpoints. Raw reports stay unscaled.
"""
import json
import matplotlib as mpl
from render_continuation_comparison import CONFIG, render
from evaluate_continuation_comparison import load_protocol
from safe_baselines_protocol import ROOT, atomic_json, require_hash, sha256


def publish():
    cfg, _, _, _, _ = load_protocol(CONFIG)
    root = ROOT / cfg["output_dir"]
    digest = sha256(root / "RESULTS.json")
    result = json.loads((root / "RESULTS.json").read_text())
    with mpl.rc_context({"axes.labelpad": 10}):
        render(CONFIG)
    t = result["teacher"]["three_seed_mean"]["pooled"]
    p = result["reconstruction"]["three_seed_mean"]["pooled"]
    a = result["source_detection"]["pooled"]["three_seed_mean"]
    deltas = {"Teacher cosine (4,000)": t["delta"], "Source AUC (2,000 held out)": a["final_minus_source"],
              "Parent-copy cosine (606)": p["copy_delta"], "Probe cosine (606)": p["probe_delta"],
              "Probe gain over copy (606)": p["gain_delta"], "Probe order gap (606)": p["order_gap_delta"]}
    table = root / "continuation_comparison_table.tex"
    lines, changed = [], set()
    for line in table.read_text().splitlines():
        fields = line.split(" & ")
        if fields[0] in deltas:
            cell = deltas[fields[0]]
            fields[3] = f"{cell['mean'] * 1e6:+.3f} [{cell['ci95'][0] * 1e6:+.3f}, {cell['ci95'][1] * 1e6:+.3f}] " + r"\\"
            line = " & ".join(fields)
            changed.add(fields[0])
        elif line.startswith("Metric (documents)"):
            line = r"Metric (documents) & Source & Final & Change [95\% CI] $(\times10^{-6})$ \\"
        elif line.startswith(r"\caption{"):
            line = line[:-1] + r" Changes and interval bounds are in units of $10^{-6}$; source/final columns are unscaled.}"
        lines.append(line)
    assert changed == set(deltas)
    table.write_text("\n".join(lines) + "\n")
    require_hash(root / "RESULTS.json", digest)
    atomic_json(root / "PRESENTATION.json", {"layout_only_after_results": True, "results_sha256": digest,
        "implementation_sha256": sha256(__file__), "frozen_renderer_sha256": sha256(ROOT / "scripts/render_continuation_comparison.py"),
        "axes_labelpad": 10, "table_change_units": 1e-6, "all_primary_outputs_unchanged": True})
    atomic_json(root / "ARTIFACT_HASHES.json", {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "ARTIFACT_HASHES.json"})


if __name__ == "__main__":
    publish()
