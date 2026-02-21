#!/bin/bash
#SBATCH -J 10_Calibration
#SBATCH -A p71186
#SBATCH -t 00:30:00
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/10_Calibration_%j.out
#SBATCH --error=results/logs/10_Calibration_%j.err

# --- SETUP ---
export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR

echo "=== STARTING TEMPERATURE CALIBRATION ==="
echo "Scanning .pt files and optimizing T for LogLoss..."

# Run the python script
python -u src/find_scaling_temperature_value.py

echo "=== CALIBRATION COMPLETE ==="