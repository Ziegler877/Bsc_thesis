#!/bin/bash
#SBATCH -J Thesis_Base_Run
#SBATCH -A p71186
#SBATCH -t 01:00:00
# --- FIX: REMOVE "-N 1" to allow shared node usage ---
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/base_run_%j.out
#SBATCH --error=results/logs/base_run_%j.err

# 1. Clean Environment & Load Modules
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

# 2. Define Project Path
PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"

# 3. Activate the Environment
source $PROJECT_DIR/.venv/bin/activate

# 4. Navigate to Project
cd $PROJECT_DIR

# Ensure logs directory exists
mkdir -p results/logs

echo "=== STARTING JOB on $(hostname) ==="
echo "Date: $(date)"
echo "Partition: zen2_0256_a40x2 (A40 GPU)"

# 5. Run The Base Models
echo "--- [1/4] Running Llama 2 (Reuters) ---"
python -u main.py --model llama2 --dataset reuters --device cuda

echo "--- [2/4] Running Llama 2 (DarkReddit) ---"
python -u main.py --model llama2 --dataset darkreddit --device cuda

echo "--- [3/4] Running Llama 4 (Reuters) ---"
python -u main.py --model llama4 --dataset reuters --device cuda

echo "--- [4/4] Running Llama 4 (DarkReddit) ---"
python -u main.py --model llama4 --dataset darkreddit --device cuda

echo "=== FINISHED ==="