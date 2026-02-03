@echo off
chcp 65001 > nul
echo STARTING LLAMA TRAINING (3 EPOCHS - BASE MODEL)

:: LLAMA 2 (7B)
echo Training Llama 2 (Reuters)...
python src/train_lora.py --model llama2 --dataset reuters --epochs 3 --batch_size 1

echo Training Llama 2 (DarkReddit)...
python src/train_lora.py --model llama2 --dataset darkreddit --epochs 3 --batch_size 1

:: LLAMA 4 (Maverick)
echo Training Llama 4 (Reuters)...
python src/train_lora.py --model llama4 --dataset reuters --epochs 3 --batch_size 1

echo Training Llama 4 (DarkReddit)...
python src/train_lora.py --model llama4 --dataset darkreddit --epochs 3 --batch_size 1

echo LLAMA TRAINING COMPLETE.
pause