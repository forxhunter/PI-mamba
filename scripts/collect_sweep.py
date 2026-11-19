"""Collect the results of a sampling sweep into one table.

Reads ``summary.json`` (scTM pipeline) and ``metrics.json`` (CA metrics) from
every subdirectory of a sweep directory and prints them side by side.

Usage:
    python scripts/collect_sweep.py --dir sweep_results/sampling
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    ("config", "{}"),
    ("n", "{:d}"),
    ("scTM", "{:.3f}"),
    ("scTM_sem", "{:.3f}"),
    ("scTM_med", "{:.3f}"),
    ("f>0.5", "{:.2f}"),
    ("f>0.7", "{:.2f}"),
    ("scRMSD", "{:.2f}"),
    ("pLDDT", "{:.1f}"),
    ("helix", "{:.3f}"),
    ("strand", "{:.3f}"),
    ("coil", "{:.3f}"),
    ("rg", "{:.2f}"),
    ("bond_dev", "{:.3f}"),
    ("clash", "{:.3f}"),
    ("diversity", "{:.3f}"),
]


def frac_above(info: Path, threshold: float) -> float:
    """Fraction of samples in info.csv with scTM above a threshold."""
    if not info.exists():
        return float("nan")
    values = []
    with open(info) as fh:
        for row in csv.DictReader(fh):
            if row.get("scTM"):
                values.append(float(row["scTM"]))
    if not values:
        return float("nan")
    return sum(v > threshold for v in values) / len(values)


def row_for(d: Path) -> dict | None:
    summary = d / "summary.json"
    metrics = d / "metrics.json"
    if not summary.exists():
        return None
    s = json.load(open(summary))
    n = int(s["n_samples"])
    row = {
        "config": d.name,
        "n": n,
        "scTM": s["scTM_mean"],
        "scTM_sem": s["scTM_std"] / max(1, n) ** 0.5,
        "scTM_med": s["scTM_median"],
        "f>0.5": s["frac_scTM_above_0.5"],
        "f>0.7": s.get("frac_scTM_above_0.7", frac_above(d / "info.csv", 0.7)),
        "scRMSD": s["scRMSD_mean"],
        "pLDDT": s["pLDDT_mean"],
    }
    if metrics.exists():
        m = json.load(open(metrics))["samples"]
        key = next(iter(m))
        m = m[key]
        row.update({
            "helix": m["helix_frac"],
            "strand": m["strand_frac"],
            "coil": m["coil_frac"],
            "rg": m["rg"],
            "bond_dev": m["bond_dev_frac"],
            "clash": m["clash_frac"],
            "diversity": m["pairwise_tm"],
        })
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    root = Path(args.dir)
    rows = [r for r in (row_for(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]
    if not rows:
        print(f"no finished configurations in {root}")
        return

    rows.sort(key=lambda r: -r["scTM"])

    names = [c for c, _ in COLUMNS]
    widths = {c: max(len(c), 9 if c != "config" else 18) for c in names}
    print("  ".join(c.rjust(widths[c]) if c != "config" else c.ljust(widths[c])
                    for c in names))
    for r in rows:
        cells = []
        for c, fmt in COLUMNS:
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                cells.append("-".rjust(widths[c]))
            elif c == "config":
                cells.append(str(v).ljust(widths[c]))
            else:
                cells.append(fmt.format(v).rjust(widths[c]))
        print("  ".join(cells))

    if args.output:
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=names)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
