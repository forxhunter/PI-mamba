#!/bin/bash
# Architecture Sweep: PI-Mamba vs Transformer vs Performer
# Generates data for Prediction P3 (Architecture Ordering)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
OUTPUT_DIR="${SCRIPT_DIR}/../sweep_results/architecture"

mkdir -p "$OUTPUT_DIR"

echo "=== Architecture Sweep ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Common settings
EPOCHS=100
BATCH_SIZE=32
LR=3e-4
SEEDS="0 1 2"

# Models to sweep
MODELS="pi_mamba transformer performer"

for MODEL in $MODELS; do
    for SEED in $SEEDS; do
        echo "Running: $MODEL seed=$SEED"
        
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_seed${SEED}.json"
        
        # Skip if already exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping (already exists)"
            continue
        fi
        
        python "${SRC_DIR}/train.py" \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --output "$OUTPUT_FILE" \
            --log_ck \
            2>&1 | tee "${OUTPUT_DIR}/${MODEL}_seed${SEED}.log"
            
        echo "  Done: $OUTPUT_FILE"
    done
done

echo ""
echo "=== Sweep Complete ==="
echo "Run analysis: python src/compute_correlation.py --input $OUTPUT_DIR"
