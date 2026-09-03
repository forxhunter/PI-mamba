#!/bin/bash
# Ablation Sweep: LR x Batch x Weight Decay
# Closes loopholes: "It's just LR", "It's just batch size", "It's just regularization"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
OUTPUT_DIR="${SCRIPT_DIR}/../sweep_results/ablation"

mkdir -p "$OUTPUT_DIR"

echo "=== Ablation Sweep ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Ablation grid
LR_VALUES="1e-4 3e-4"
BATCH_SIZES="32 64"
WD_VALUES="0 1e-4"
SEED=42

for LR in $LR_VALUES; do
    for BS in $BATCH_SIZES; do
        for WD in $WD_VALUES; do
            CONFIG="lr${LR}_bs${BS}_wd${WD}"
            echo "Running: $CONFIG"
            
            OUTPUT_FILE="${OUTPUT_DIR}/${CONFIG}.json"
            
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  Skipping (already exists)"
                continue
            fi
            
            python "${SRC_DIR}/train.py" \
                --model pi_mamba \
                --epochs 50 \
                --batch_size "$BS" \
                --lr "$LR" \
                --weight_decay "$WD" \
                --seed "$SEED" \
                --output "$OUTPUT_FILE" \
                --log_ck \
                2>&1 | tee "${OUTPUT_DIR}/${CONFIG}.log"
                
            echo "  Done"
        done
    done
done

echo ""
echo "=== Ablation Sweep Complete ==="
