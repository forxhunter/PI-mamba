#!/usr/bin/env bash
# Full beta-sheet readout for a control checkpoint and one or more arms.
#
# Every checkpoint goes through the same four measurements, in the same order,
# from the same script, so that nothing differs between the control and an arm
# except the weights:
#
#   C_10            fraction of hidden-state variance in the 10 slowest Rouse
#                   modes, the mechanism variable of the representation
#                   hypothesis (measure_mode_concentration)
#   from noise      sheet content of samples generated from Gaussian noise,
#                   relaxed and projection-only. This is the quantity that
#                   matters, and the one with the least resolution: at 300
#                   samples registered_frac reads about 0.01 either way.
#   retention       register kept when the model is handed a noised training
#                   structure at t0 and integrates to t=1 (sheet_recovery).
#                   Paired structure by structure against the control, which
#                   is what gives this measurement resolution (sem ~0.01).
#
# The paired retention comparison is the primary readout. The from-noise
# numbers are reported because they are the goal, not because a difference of
# 0.005 in them means anything.
#
# Usage:
#   scripts/arm_readout.sh control=ckpt.pt scrambled=ckpt.pt learnedA=ckpt.pt
#
# The first argument is the control; every later one is compared against it.
# Environment: OUT, N_NOISE, N_RECOV, LENGTH, STEPS, T0, DATA.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 control=<ckpt> <name>=<ckpt> [<name>=<ckpt> ...]" >&2
  exit 2
fi

OUT=${OUT:-eval_output/arms}
N_NOISE=${N_NOISE:-300}
N_RECOV=${N_RECOV:-200}
LENGTH=${LENGTH:-100}
STEPS=${STEPS:-200}
T0=${T0:-0.9}
DATA=${DATA:-Distilled_trainset}

mkdir -p "$OUT"
names=()

for spec in "$@"; do
  case "$spec" in
    *=*) ;;
    *) echo "expected name=checkpoint, got '$spec'" >&2; exit 2 ;;
  esac
  name=${spec%%=*}
  ckpt=${spec#*=}
  if [ ! -f "$ckpt" ]; then
    echo "no such checkpoint: $ckpt" >&2
    exit 2
  fi
  names+=("$name")

  # SUMMARY_ONLY=1 reprints the tables from a finished run's JSON files
  # without measuring anything again.
  if [ -n "${SUMMARY_ONLY:-}" ]; then
    continue
  fi

  echo "############ $name: $ckpt"

  echo "==== $name: C_10"
  python -m src.measure_mode_concentration \
    --checkpoint "$ckpt" --data_dir "$DATA" --k 10 \
    --output "$OUT/ck_${name}.json"

  for relax in relax norelax; do
    flag=""
    [ "$relax" = norelax ] && flag="--no_relax"
    echo "==== $name: from noise, $relax"
    python scripts/evaluate.py --mode generate \
      --checkpoint "$ckpt" --n_samples "$N_NOISE" --length "$LENGTH" \
      --n_steps "$STEPS" $flag --eval_dir "$OUT/noise_${name}_${relax}"
    python scripts/sheet_quality.py \
      --samples "$OUT/noise_${name}_${relax}/pdbs" \
      --length "$LENGTH" \
      --output "$OUT/sheet_${name}_${relax}.json"
  done

  echo "==== $name: register retention at t0=$T0"
  python scripts/sheet_recovery.py \
    --checkpoint "$ckpt" --reference "$DATA" \
    --n "$N_RECOV" --t0 "$T0" \
    --output "$OUT/recov_${name}.json"
done

control=${names[0]}
for name in "${names[@]:1}"; do
  echo
  echo "############ paired retention: $control -> $name"
  python scripts/compare_recovery.py \
    "$OUT/recov_${control}.json" "$OUT/recov_${name}.json" \
    --key "t0=$T0" || echo "(no paired comparison for $name)"
done

echo
echo "############ C_10"
python - "$OUT" "${names[@]}" <<'EOF'
import json
import sys

out, names = sys.argv[1], sys.argv[2:]
print(f"{'arm':<14}{'C_10':>10}")
for name in names:
    with open(f"{out}/ck_{name}.json") as fh:
        d = json.load(fh)
    value = d.get("C_k", d.get("C_10"))
    print(f"{name:<14}{value:>10.4f}" if isinstance(value, (int, float))
          else f"{name:<14}{str(value):>10}")
EOF

echo
echo "############ from noise"
python - "$OUT" "${names[@]}" <<'EOF'
import json
import math
import sys

out, names = sys.argv[1], sys.argv[2:]
fields = ("strand_frac", "paired_strand_frac", "registered_frac", "n_ladders")

# sheet_quality.py keys its output by the name of the sample directory, with
# <field> and <field>_std inside. There is one such entry per file here.
for relax in ("relax", "norelax"):
    print(f"\n{relax}  (mean +- sem over n samples)")
    print(f"{'arm':<12}{'n':>5}" + "".join(f"{f[:16]:>18}" for f in fields))
    for name in names:
        with open(f"{out}/sheet_{name}_{relax}.json") as fh:
            entries = json.load(fh)
        stats = next(iter(entries.values())) if entries else {}
        n = stats.get("n", 0)
        row = f"{name:<12}{n:>5}"
        for f in fields:
            mean, std = stats.get(f), stats.get(f + "_std")
            if not isinstance(mean, (int, float)) or math.isnan(mean):
                row += f"{'-':>18}"
                continue
            sem = (std / math.sqrt(n)) if (isinstance(std, (int, float))
                                          and not math.isnan(std) and n) else float("nan")
            row += f"{mean:>11.3f}+-{sem:<5.3f}"
        print(row)
EOF
