#!/bin/bash
# Monitor V14 training and run eval when done
cd /data2/2026_RNAAI/FINAL_SUBMISSION/code/scripts

while true; do
    # Check if training is still running
    if ! pgrep -f "train.py.*checkpoints_v14" > /dev/null 2>&1; then
        echo "$(date): Training finished! Starting evaluation..."
        
        # Step 1: Generate samples
        echo "$(date): Generating 100 samples at L=100..."
        CUDA_VISIBLE_DEVICES=0 conda run -n fm python evaluate.py \
            --mode generate \
            --checkpoint ../checkpoints_v14/best_model.pt \
            --n_samples 100 --length 100 --n_steps 200 \
            --eval_dir /data2/2026_RNAAI/FINAL_SUBMISSION/code/eval_output_v14
        
        echo "$(date): Generation done. Starting scTM evaluation..."
        
        # Step 2: Run scTM pipeline
        CUDA_VISIBLE_DEVICES=0 conda run -n Proteus python evaluate.py \
            --mode eval \
            --eval_dir /data2/2026_RNAAI/FINAL_SUBMISSION/code/eval_output_v14
        
        echo "$(date): Evaluation complete!"
        echo "Results:"
        cat /data2/2026_RNAAI/FINAL_SUBMISSION/code/eval_output_v14/info.csv 2>/dev/null || echo "Check eval_output_v14 for results"
        break
    fi
    
    echo "$(date): Training still running..."
    sleep 600  # Check every 10 minutes
done
