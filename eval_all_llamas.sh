#!/bin/bash
#SBATCH --job-name=eval_all_llamas
#SBATCH -A p71186
#SBATCH --output=logs/eval_all_llamas_%j.out
#SBATCH --error=logs/eval_all_llamas_%j.err
#SBATCH -t 12:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2

# Go to your project root
cd /gpfs/data/fs71186/ziegler/ThesisProject

# Set your pooling strategy here (e.g., "mean", "gmp", or "dynamic")
POOLING="mean"

echo "========================================================="
echo " STARTING SEQUENTIAL EVALUATION FOR ALL LLAMA MODELS"
echo " Pooling Strategy: $POOLING"
echo "========================================================="
echo ""

# Loop through models
for MODEL in llama2 llama3; do
    # Loop through datasets
    for DATASET in darkreddit reuters; do
        # Loop through runs 1 to 5
        for RUN in {1..5}; do

            echo "---------------------------------------------------------"
            echo " -> Evaluating: Model=$MODEL | Dataset=$DATASET | Run=$RUN"
            echo "---------------------------------------------------------"

            python -u main.py \
                --model $MODEL \
                --dataset $DATASET \
                --lora \
                --pooling $POOLING \
                --suffix "_run${RUN}"

            echo " -> Finished: ${MODEL} | ${DATASET} | _run${RUN}"
            echo ""

        done
    done
done

echo "========================================================="
echo " ALL 20 EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "========================================================="