#!/usr/bin/env python3
"""Checks for scripts/compare_trajectories.py.

Run directly: python tests/test_compare_trajectories.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_trajectories as ct  # noqa: E402


def write_log(path: Path, records) -> Path:
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def val(step: int, loss: float, strand: float) -> dict:
    return {
        "type": "val",
        "step": step,
        "val_loss": loss,
        "samples": {"L100": {"strand_frac": strand}},
    }


def test_dig_follows_a_dotted_path():
    rec = val(1000, 0.5, 0.1)
    assert ct.dig(rec, "val_loss") == 0.5
    assert ct.dig(rec, "samples.L100.strand_frac") == 0.1
    assert ct.dig(rec, "samples.L200.strand_frac") is None, "missing key is None"
    assert ct.dig(rec, "val_loss.deeper") is None, "cannot index into a float"
    assert ct.dig(rec, "samples") is None, "a dict is not a number"
    print("  dotted paths and missing keys behave")


def test_only_val_records_are_read_and_resumes_are_deduplicated():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_log(Path(tmp) / "log.jsonl", [
            {"type": "train", "step": 500, "loss": 1.0},
            val(1000, 0.60, 0.01),
            val(2000, 0.55, 0.02),
            {"type": "train", "step": 2500, "loss": 0.9},
            # A resumed run writes step 2000 again; the later record wins.
            val(2000, 0.50, 0.05),
            val(3000, 0.45, 0.07),
        ])
        recs = ct.read_val_records(path)
    steps = [r["step"] for r in recs]
    assert steps == [1000, 2000, 3000], steps
    assert recs[1]["val_loss"] == 0.50, "the later record for a step replaces the earlier"
    print(f"  read {len(recs)} val records, resume duplicate resolved")


def test_a_truncated_last_line_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        with path.open("w") as fh:
            fh.write(json.dumps(val(1000, 0.6, 0.01)) + "\n")
            fh.write('{"type": "val", "step": 2000, "val_los')  # still being written
        recs = ct.read_val_records(path)
    assert [r["step"] for r in recs] == [1000], "a live run's partial line is skipped"
    print("  partial last line of a live log is skipped")


def test_a_shorter_run_reports_nothing_past_its_end():
    """The bug this pins: carrying a finished run's last value forward.

    A run that stopped at 2000 steps must not appear as a flat line next to a
    run that reached 4000, because that reads as 'the arm stayed level' when
    the arm was simply not there.
    """
    short = [val(1000, 0.60, 0.01), val(2000, 0.55, 0.02)]
    long = [val(1000, 0.60, 0.01), val(2000, 0.55, 0.02),
            val(3000, 0.50, 0.03), val(4000, 0.45, 0.04)]
    for step in (3000, 4000):
        got, n = ct.windowed(short, step, "val_loss", 0)
        assert got is None and n == 0, f"short run reported {got} at step {step}"
    got, _ = ct.windowed(long, 4000, "val_loss", 0)
    assert got == 0.45, got
    got, _ = ct.windowed(short, 2000, "val_loss", 0)
    assert got == 0.55, got
    print("  a run reports only up to the step it reached")


def test_window_averages_a_span_of_steps():
    recs = [val(1000, 0.6, 0.00), val(2000, 0.5, 0.02),
            val(3000, 0.4, 0.04), val(4000, 0.3, 0.06)]
    got, n = ct.windowed(recs, 4000, "samples.L100.strand_frac", 0)
    assert abs(got - 0.06) < 1e-9 and n == 1, (got, n)
    got, n = ct.windowed(recs, 4000, "samples.L100.strand_frac", 2000)
    assert abs(got - (0.02 + 0.04 + 0.06) / 3) < 1e-9 and n == 3, (got, n)
    # A window reaching past the start of the run averages what there is.
    got, n = ct.windowed(recs, 2000, "samples.L100.strand_frac", 50000)
    assert abs(got - 0.01) < 1e-9 and n == 2, (got, n)
    print("  window averages the records inside a span of steps")


def test_two_validation_intervals_average_the_same_span():
    """The bias this pins: a window counted in records, not steps.

    One run validates every 1000 steps and the other every 2500. Both follow
    the same rising curve, so a fair comparison must report no difference. A
    window of five records would reach back 5000 steps in the first run and
    12500 in the second, and the coarser run would read low.
    """
    def curve(step: int) -> float:
        return 0.02 + 1e-6 * step          # rises by 0.01 per 10000 steps

    fine = [val(s, 0.5, curve(s)) for s in range(1000, 50001, 1000)]
    coarse = [val(s, 0.5, curve(s)) for s in range(2500, 50001, 2500)]

    a, na = ct.windowed(fine, 50000, "samples.L100.strand_frac", 10000)
    b, nb = ct.windowed(coarse, 50000, "samples.L100.strand_frac", 10000)
    # The window is inclusive at both ends, so both runs take their record at
    # step 40000: 11 records at an interval of 1000, 5 at 2500.
    assert na == 11 and nb == 5, (na, nb)
    assert abs(a - b) < 1e-6, f"same span gave {a:.6f} and {b:.6f}"
    print(f"  {na} and {nb} records over the same 10000 steps agree to "
          f"{abs(a - b):.2e}")


def test_main_prints_a_table_for_two_runs():
    with tempfile.TemporaryDirectory() as tmp:
        a = write_log(Path(tmp) / "a.jsonl",
                      [val(s, 0.6 - s / 1e5, 0.01 * s / 1000) for s in range(1000, 5001, 1000)])
        b = write_log(Path(tmp) / "b.jsonl",
                      [val(s, 0.6 - s / 9e4, 0.02 * s / 1000) for s in range(1000, 5001, 1000)])
        code = ct.main(["--log", f"ctrl={a}", "--log", f"arm={b}",
                        "--field", "samples.L100.strand_frac",
                        "--every", "2000", "--window", "2000"])
    assert code == 0
    print("  main() runs over two logs")


def main() -> int:
    checks = [
        test_dig_follows_a_dotted_path,
        test_only_val_records_are_read_and_resumes_are_deduplicated,
        test_a_truncated_last_line_is_skipped,
        test_a_shorter_run_reports_nothing_past_its_end,
        test_window_averages_a_span_of_steps,
        test_two_validation_intervals_average_the_same_span,
        test_main_prints_a_table_for_two_runs,
    ]
    passed = 0
    for check in checks:
        print(f"{check.__name__}")
        check()
        passed += 1
    print(f"\n{passed} of {len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
