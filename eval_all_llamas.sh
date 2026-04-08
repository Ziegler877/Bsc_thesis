#!/bin/bash
#SBATCH -J 926_ldval_all_llamas
#SBATCH -A p71186
#SBATCH -t 12:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:2
#SBATCH --output=logs/eval_all_llamas_%j.out
#SBATCH --error=logs/eval_all_llamas_%j.err

export WANDB_API_KEY=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
export WANDB_PROJECT="BSC Thesis"
module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load cuda/11.8.0-gcc-11.2.0-411

PROJECT_DIR="/gpfs/data/fs71186/ziegler/ThesisProject"
source $PROJECT_DIR/.venv/bin/activate
cd $PROJECT_DIR


echo "========================================================="
echo " STARTING FULL SEQUENTIAL EVALUATION (NO LOOPS)"
echo "========================================================="

# ---------------------------------------------------------
# LLAMA 2 - DARKREDDIT (Pooling: mean)
# ---------------------------------------------------------
echo "Evaluating: Llama 2 | DarkReddit | Run 1"
python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean --suffix "_run1"

echo "Evaluating: Llama 2 | DarkReddit | Run 2"
python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean --suffix "_run2"

echo "Evaluating: Llama 2 | DarkReddit | Run 3"
python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean --suffix "_run3"

echo "Evaluating: Llama 2 | DarkReddit | Run 4"
python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean --suffix "_run4"

echo "Evaluating: Llama 2 | DarkReddit | Run 5"
python -u main.py --model llama2 --dataset darkreddit --lora --pooling mean --suffix "_run5"


# ---------------------------------------------------------
# LLAMA 2 - REUTERS (Pooling: mean)
# ---------------------------------------------------------
echo "Evaluating: Llama 2 | Reuters | Run 1"
python -u main.py --model llama2 --dataset reuters --lora --pooling mean --suffix "_run1"

echo "Evaluating: Llama 2 | Reuters | Run 2"
python -u main.py --model llama2 --dataset reuters --lora --pooling mean --suffix "_run2"

echo "Evaluating: Llama 2 | Reuters | Run 3"
python -u main.py --model llama2 --dataset reuters --lora --pooling mean --suffix "_run3"

echo "Evaluating: Llama 2 | Reuters | Run 4"
python -u main.py --model llama2 --dataset reuters --lora --pooling mean --suffix "_run4"

echo "Evaluating: Llama 2 | Reuters | Run 5"
python -u main.py --model llama2 --dataset reuters --lora --pooling mean --suffix "_run5"


# ---------------------------------------------------------
# LLAMA 3 - DARKREDDIT (Pooling: dynamic)
# ---------------------------------------------------------
echo "Evaluating: Llama 3 | DarkReddit | Run 1"
python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic --suffix "_run1"

echo "Evaluating: Llama 3 | DarkReddit | Run 2"
python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic --suffix "_run2"

echo "Evaluating: Llama 3 | DarkReddit | Run 3"
python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic --suffix "_run3"

echo "Evaluating: Llama 3 | DarkReddit | Run 4"
python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic --suffix "_run4"

echo "Evaluating: Llama 3 | DarkReddit | Run 5"
python -u main.py --model llama3 --dataset darkreddit --lora --pooling dynamic --suffix "_run5"


# ---------------------------------------------------------
# LLAMA 3 - REUTERS (Pooling: mean)
# ---------------------------------------------------------
echo "Evaluating: Llama 3 | Reuters | Run 1"
python -u main.py --model llama3 --dataset reuters --lora --pooling mean --suffix "_run1"

echo "Evaluating: Llama 3 | Reuters | Run 2"
python -u main.py --model llama3 --dataset reuters --lora --pooling mean --suffix "_run2"

echo "Evaluating: Llama 3 | Reuters | Run 3"
python -u main.py --model llama3 --dataset reuters --lora --pooling mean --suffix "_run3"

echo "Evaluating: Llama 3 | Reuters | Run 4"
python -u main.py --model llama3 --dataset reuters --lora --pooling mean --suffix "_run4"

echo "Evaluating: Llama 3 | Reuters | Run 5"
python -u main.py --model llama3 --dataset reuters --lora --pooling mean --suffix "_run5"

echo "========================================================="
echo " ALL 20 EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "========================================================="