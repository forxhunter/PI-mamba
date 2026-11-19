#!/usr/bin/env bash
# Sampling sweep: which sampler settings change scTM, without retraining.
#
# Each configuration generates --n samples of length 100 from the same
# checkpoint, then runs CA metrics and the full scTM pipeline on them. The
# diagnosis script is run as well, so that the secondary-structure breakdown of
# every configuration is on record. scTM alone can be raised by collapsing the
# output onto helical bundles, so the sweep is only interpretable together with
# helix_frac, strand_frac and pairwise_tm.
#
# Usage (from the repository root, inside a gpuq slot):
#   gpuq run --mem 13G --ncpu 4 --name sweep -- bash scripts/sweep_sampling.sh
#
# Environment overrides: N (samples per configuration), SEED, OUT, CKPT.

set -u

N=${N:-100}
SEED=${SEED:-1}
OUT=${OUT:-sweep_results/sampling}
CKPT=${CKPT:-checkpoints/paper_512x16/best_model.pt}
LAST=${LAST:-checkpoints/paper_512x16/last.pt}
# The scTM step needs an interpreter with transformers installed, a ProteinMPNN
# checkout and an ESMFold snapshot. Set these for your machine.
VESM=${VESM:-python}
MPNN=${MPNN:-${PROTEINMPNN_DIR:-}}
ESMFOLD=${ESMFOLD:-facebook/esmfold_v1}

if [ -z "$MPNN" ]; then
  echo "set MPNN or PROTEINMPNN_DIR to a ProteinMPNN checkout" >&2
  exit 2
fi

mkdir -p "$OUT"

# name|checkpoint|seed|extra generate flags
CONFIGS=(
  "base200|$CKPT|$SEED|--n_steps 200"
  "base200_seed2|$CKPT|2|--n_steps 200"
  "steps50|$CKPT|$SEED|--n_steps 50"
  "steps100|$CKPT|$SEED|--n_steps 100"
  "steps500|$CKPT|$SEED|--n_steps 500"
  "steps1000|$CKPT|$SEED|--n_steps 1000"
  "steps2000|$CKPT|$SEED|--n_steps 2000"
  "raw_weights|$CKPT|$SEED|--n_steps 200 --no_ema"
  "last_ckpt|$LAST|$SEED|--n_steps 200"
  "no_relax|$CKPT|$SEED|--n_steps 200 --no_relax"
  "no_projection|$CKPT|$SEED|--n_steps 200 --no_projection"
  "steps1000_norelax|$CKPT|$SEED|--n_steps 1000 --no_relax"
  "steps1000_raw|$CKPT|$SEED|--n_steps 1000 --no_ema"
)

# ARMS, if set, is a comma-separated list of configuration names to run; the
# rest are skipped. Use it to run the informative arms first.
ARMS=${ARMS:-}

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name ckpt seed flags <<< "$entry"
  if [ -n "$ARMS" ] && [[ ",$ARMS," != *",$name,"* ]]; then
    continue
  fi
  dir="$OUT/$name"

  if [ -f "$dir/summary.json" ]; then
    echo "=== $name: already done, skipping"
    continue
  fi

  echo "=== $name  ($flags, checkpoint $(basename "$ckpt"), seed $seed)"
  rm -rf "$dir"
  mkdir -p "$dir"

  # shellcheck disable=SC2086
  python scripts/evaluate.py --mode generate \
      --checkpoint "$ckpt" --n_samples "$N" --length 100 \
      --eval_dir "$dir" --seed "$seed" $flags \
      > "$dir/generate.log" 2>&1 || { echo "  generate failed, see $dir/generate.log"; continue; }

  python scripts/evaluate.py --mode metrics --eval_dir "$dir" \
      > "$dir/metrics.log" 2>&1 || echo "  metrics failed"

  "$VESM" scripts/evaluate.py --mode eval --eval_dir "$dir" \
      --proteinmpnn_dir "$MPNN" --esmfold_model "$ESMFOLD" --seed 0 \
      > "$dir/sctm.log" 2>&1 || { echo "  scTM failed, see $dir/sctm.log"; continue; }

  python scripts/diagnose_sctm.py --eval_dir "$dir" \
      > "$dir/diagnosis.log" 2>&1 || echo "  diagnosis failed"

  python - "$dir" <<'PY'
import json, sys
d = sys.argv[1]
s = json.load(open(f"{d}/summary.json"))
m = json.load(open(f"{d}/metrics.json"))["samples"]["L100"]
print(f"  scTM {s['scTM_mean']:.3f} +- {s['scTM_std']:.3f}, "
      f">0.5 {s['frac_scTM_above_0.5']:.2f}, pLDDT {s['pLDDT_mean']:.1f}, "
      f"helix {m['helix_frac']:.3f}, strand {m['strand_frac']:.3f}, "
      f"div {m['pairwise_tm']:.3f}, bond_dev {m['bond_dev_frac']:.3f}")
PY
done

echo
echo "=== sweep table"
python scripts/collect_sweep.py --dir "$OUT" --output "$OUT/table.csv"
