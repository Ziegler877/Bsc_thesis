@echo off
chcp 65001 > nul
echo STARTING TRAINING (5 EPOCHS)

:: E5 (Safe)
python src/train_lora.py --model e5_small --dataset reuters --epochs 5 --batch_size 16
python src/train_lora.py --model e5_small --dataset darkreddit --epochs 5 --batch_size 16
python src/train_lora.py --model e5_large --dataset reuters --epochs 5 --batch_size 4
python src/train_lora.py --model e5_large --dataset darkreddit --epochs 5 --batch_size 4


echo TRAINING COMPLETE.
pause