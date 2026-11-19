#!/bin/bash
# Architecture Sweep: PI-Mamba vs Transformer vs Performer
# Generates data for Prediction P3 (Architecture Ordering)
#
# Extra arguments are forwarded to scripts/train.py, for example:
#   EPOCHS=20 bash scripts/run_architecture_sweep.sh --d_model 256 --n_layers 8
#
# OUTPUT_DIR, MODELS, SEEDS, BATCH_SIZE and LR can be set in the environment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../sweep_results/architecture}"

mkdir -p "$OUTPUT_DIR"

echo "=== Architecture Sweep ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Common settings
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
SEEDS="${SEEDS:-0 1 2}"

# Models to sweep
MODELS="${MODELS:-pi_mamba transformer performer}"

for MODEL in $MODELS; do
    for SEED in $SEEDS; do
        echo "Running: $MODEL seed=$SEED"
        
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_seed${SEED}.json"
        
        # Skip if already exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping (already exists)"
            continue
        fi
        
        python "${SCRIPT_DIR}/train.py" \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --checkpoint_dir "${OUTPUT_DIR}/${MODEL}_seed${SEED}" \
            --output "$OUTPUT_FILE" \
            --log_ck \
            "$@" \
            2>&1 | tee "${OUTPUT_DIR}/${MODEL}_seed${SEED}.log"
            
        echo "  Done: $OUTPUT_FILE"
    done
done

echo ""
echo "=== Sweep Complete ==="
echo "Run analysis: python ${SCRIPT_DIR}/../src/compute_correlation.py --input $OUTPUT_DIR"
