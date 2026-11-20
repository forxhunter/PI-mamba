#!/usr/bin/env bash
# Compare two training arms at the same step on register retention.
#
# Both arms continue from the same checkpoint on the same learning-rate
# schedule, so a checkpoint of each at the same step differs only by the loss
# they were trained with. sheet_recovery.py noises the same reference
# structures with the same seed for both, so the retention can be compared
# structure by structure.
#
# Usage:
#   scripts/compare_arms.sh <baseline.pt> <candidate.pt> <tag> [n]
set -euo pipefail

BASE=${1:?baseline checkpoint}
CAND=${2:?candidate checkpoint}
TAG=${3:?tag for the output files}
N=${4:-200}
T0=${T0:-0.9}
OUT=${OUT:-eval_output}

mkdir -p "$OUT"
for pair in "base:$BASE" "cand:$CAND"; do
  name=${pair%%:*}
  ckpt=${pair#*:}
  echo "=== $name: $ckpt"
  python scripts/sheet_recovery.py \
    --checkpoint "$ckpt" --reference Distilled_trainset \
    --n "$N" --t0 "$T0" \
    --output "$OUT/recov_${TAG}_${name}.json"
done

echo
python scripts/compare_recovery.py \
  "$OUT/recov_${TAG}_base.json" "$OUT/recov_${TAG}_cand.json" \
  --key "t0=$T0"
