#!/usr/bin/env python3
"""Compare validation trajectories of several training runs.

Each run writes one JSON object per line to ``train_log.jsonl``, including a
``type: "val"`` record with the validation loss, the Rouse mode concentration
and the secondary-structure composition of a small batch of samples. This
script lines those records up by step so that two runs can be read against
each other over the whole of training rather than at one point.

Reading one point is what made an earlier pair of ablation arms useless: both
were measured at 20000 steps, where neither had produced a beta ladder yet, so
both read zero and the comparison carried no information. Ladders appear
somewhere between 20000 and 100000 steps, so an arm has to be followed.

Each validation record samples 16 structures per length, so a single
``strand_frac`` reading is noisy. ``--window`` averages over the last k
validation records at or before each reported step, which is what makes a
difference of a few percent legible.

Usage:
    python scripts/compare_trajectories.py \
        --log control=checkpoints/paper_512x16/train_log.jsonl \
        --log scrambled=checkpoints/arm_scrambled_512x16/train_log.jsonl \
        --field samples.L100.strand_frac --field val_loss \
        --every 10000 --window 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True, metavar="NAME=PATH",
                   help="a run to read, as name=path/to/train_log.jsonl (repeatable); "
                        "the first one given is the baseline for the diff column")
    p.add_argument("--field", action="append", metavar="DOTTED.PATH",
                   help="a field of the val record, e.g. val_loss or "
                        "samples.L100.strand_frac (repeatable)")
    p.add_argument("--every", type=int, default=10000,
                   help="report at multiples of this many steps (default 10000)")
    p.add_argument("--window", type=int, default=1,
                   help="average over the last k val records up to each reported "
                        "step (default 1, no averaging)")
    p.add_argument("--last", action="store_true",
                   help="also report each run's most recent val record, even if "
                        "it is not a multiple of --every")
    return p.parse_args(argv)


DEFAULT_FIELDS = (
    "val_loss",
    "C_10",
    "samples.L100.helix_frac",
    "samples.L100.strand_frac",
    "samples.L200.strand_frac",
)


def dig(record: dict, dotted: str) -> Optional[float]:
    """Follow a dotted path into nested dicts; None if any step is missing."""
    node = record
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else None


def read_val_records(path: Path) -> List[dict]:
    """Every ``type: "val"`` record of a run, in step order.

    A resumed run writes the steps it repeats again, so the later record for a
    step replaces the earlier one.
    """
    by_step: Dict[int, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written last line of a live run
            if rec.get("type") == "val" and "step" in rec:
                by_step[int(rec["step"])] = rec
    return [by_step[s] for s in sorted(by_step)]


def windowed(records: Sequence[dict], step: int, field: str,
             window: int) -> Tuple[Optional[float], int]:
    """Mean of ``field`` over the last ``window`` val records up to ``step``.

    A run that has not reached ``step`` reports nothing rather than its last
    value, so that a shorter run does not appear as a flat line next to a
    longer one.
    """
    if not records or records[-1]["step"] < step:
        return None, 0
    vals = [v for r in records if r["step"] <= step
            for v in [dig(r, field)] if v is not None]
    if not vals:
        return None, 0
    take = vals[-max(window, 1):]
    return sum(take) / len(take), len(take)


def fmt(value: Optional[float]) -> str:
    return "     -" if value is None else f"{value:6.3f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    runs: List[Tuple[str, List[dict]]] = []
    for spec in args.log:
        if "=" not in spec:
            raise SystemExit(f"--log wants name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        records = read_val_records(Path(path))
        if not records:
            print(f"warning: no val records in {path}")
        runs.append((name, records))

    fields = tuple(args.field) if args.field else DEFAULT_FIELDS
    last_steps = {name: (recs[-1]["step"] if recs else 0) for name, recs in runs}
    horizon = max(last_steps.values(), default=0)
    steps = list(range(args.every, horizon + 1, args.every))
    if args.last:
        steps = sorted(set(steps) | set(last_steps.values()))

    print(f"window {args.window} val record(s); {len(runs)} run(s)")
    for name, recs in runs:
        print(f"  {name:<14} {len(recs):>4} val records, last step {last_steps[name]}")
    base_name = runs[0][0]

    for field in fields:
        print(f"\n{field}")
        header = f"{'step':>8}" + "".join(f"{n[:11]:>12}" for n, _ in runs)
        if len(runs) > 1:
            header += "".join(f"{('d:' + n)[:11]:>12}" for n, _ in runs[1:])
        print(header)
        for step in steps:
            vals = [windowed(recs, step, field, args.window)[0] for _, recs in runs]
            if all(v is None for v in vals):
                continue
            row = f"{step:>8}" + "".join(f"{fmt(v):>12}" for v in vals)
            if len(runs) > 1:
                base = vals[0]
                for v in vals[1:]:
                    d = None if (v is None or base is None) else v - base
                    row += f"{'           -' if d is None else f'{d:+12.3f}'}"
            print(row)
        if len(runs) > 1:
            print(f"         d: columns are each run minus {base_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
