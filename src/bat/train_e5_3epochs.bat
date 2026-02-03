@echo off
chcp 65001 > nul
echo STARTING TRAINING (3 EPOCHS)

:: E5 (Safe)
python src/train_lora.py --model e5_small --dataset reuters --epochs 3 --batch_size 16
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 3 --batch_size 16
python src/train_lora.py --model e5_large --dataset reuters --epochs 3 --batch_size 4
python src/train_lora.py --model e5_large --dataset darkreddit --epochs 3 --batch_size 4

echo TRAINING COMPLETE.
pause