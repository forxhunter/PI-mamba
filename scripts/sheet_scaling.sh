#!/usr/bin/env bash
# Does the ability to generate real beta sheets grow with model size and
# training steps?
#
# Generates samples from every checkpoint given and measures sheet pairing with
# scripts/sheet_quality.py. scTM is not needed: the question is whether paired,
# in-register strands appear at all, and that is measured from the CA trace.
#
# Usage (inside a gpuq slot):
#   gpuq run --mem 4G --ncpu 2 --name sheetscale -- bash scripts/sheet_scaling.sh
#
# Environment overrides: N, SEED, OUT.

set -u

N=${N:-100}
SEED=${SEED:-1}
OUT=${OUT:-sweep_results/sheet_scaling}
STEPS=${STEPS:-200}

mkdir -p "$OUT"

# label|checkpoint. CKPT_LIST overrides it with the same "label|path" entries,
# separated by spaces, so that an arm can be measured as soon as it finishes.
if [ -n "${CKPT_LIST:-}" ]; then
  read -r -a CKPTS <<< "$CKPT_LIST"
else
  CKPTS=(
    "probe_256x8_4k|checkpoints/probe_v2/best_model.pt"
    "small_256x8_20k|checkpoints/pimamba_256x8_v2/best_model.pt"
    "paper_512x16_100k|checkpoints/paper_512x16/best_model.pt"
  )
fi

dirs=()
for entry in "${CKPTS[@]}"; do
  IFS='|' read -r label ckpt <<< "$entry"
  dir="$OUT/$label"
  dirs+=("$dir/pdbs")

  if [ -d "$dir/pdbs" ] && [ "$(find "$dir/pdbs" -name '*.pdb' | wc -l)" -ge "$N" ]; then
    echo "=== $label: samples present, reusing"
    continue
  fi
  if [ ! -f "$ckpt" ]; then
    echo "=== $label: no checkpoint at $ckpt, skipping"
    continue
  fi

  echo "=== $label: generating $N samples from $ckpt"
  rm -rf "$dir"
  mkdir -p "$dir"
  python scripts/evaluate.py --mode generate --checkpoint "$ckpt" \
      --n_samples "$N" --length 100 --eval_dir "$dir" --seed "$SEED" \
      --n_steps "$STEPS" > "$dir/generate.log" 2>&1 \
    || { echo "  generate failed, see $dir/generate.log"; continue; }
  python scripts/evaluate.py --mode metrics --eval_dir "$dir" \
      > "$dir/metrics.log" 2>&1 || echo "  metrics failed"
done

echo
echo "=== sheet pairing across checkpoints, with the training set as the target"
args=()
for d in "${dirs[@]}"; do
  [ -d "$d" ] && args+=(--samples "$d")
done
python scripts/sheet_quality.py "${args[@]}" \
    --samples Distilled_trainset --samples PImamba_L100 \
    --length 100 --limit 300 --output "$OUT/sheet_quality.json"
