#!/bin/bash
# Wait for a training run to finish, then generate samples and evaluate them.
#
# Usage:
#   bash scripts/monitor_and_eval.sh <checkpoint_dir> [eval_dir]
#
# Environment:
#   PROTEINMPNN_DIR  ProteinMPNN checkout; if unset, the scTM step is skipped
#   N_SAMPLES (100), LENGTH (100), N_STEPS (200), POLL_SECONDS (600)

set -e

CKPT_DIR="${1:?usage: monitor_and_eval.sh <checkpoint_dir> [eval_dir]}"
EVAL_DIR="${2:-${CKPT_DIR}/eval_output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Wait while a train.py process that writes to this checkpoint directory is alive
while pgrep -f "train.py.*${CKPT_DIR}" > /dev/null 2>&1; do
    echo "$(date): Training still running..."
    sleep "${POLL_SECONDS:-600}"
done
echo "$(date): Training finished. Starting evaluation..."

echo "$(date): Generating ${N_SAMPLES:-100} samples at L=${LENGTH:-100}..."
python "${SCRIPT_DIR}/evaluate.py" --mode generate \
    --checkpoint "${CKPT_DIR}/best_model.pt" \
    --n_samples "${N_SAMPLES:-100}" --length "${LENGTH:-100}" --n_steps "${N_STEPS:-200}" \
    --eval_dir "${EVAL_DIR}"

echo "$(date): Computing structure metrics..."
python "${SCRIPT_DIR}/evaluate.py" --mode metrics --eval_dir "${EVAL_DIR}"

if [ -n "${PROTEINMPNN_DIR}" ]; then
    echo "$(date): Starting scTM evaluation..."
    python "${SCRIPT_DIR}/evaluate.py" --mode eval --eval_dir "${EVAL_DIR}"
    echo "Results:"
    cat "${EVAL_DIR}/summary.json"
else
    echo "PROTEINMPNN_DIR is not set; skipping the scTM evaluation."
fi

echo "$(date): Evaluation complete."
