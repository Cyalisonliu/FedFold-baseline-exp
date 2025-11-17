# FedFold ,Trainable width ratio = 16:1 or 8:1 (required: 16:1, n_split = 2, 4, 6)
# random quantize bits
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 2 --train_ratio 16-1 --quantize
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 2 --train_ratio 16-1 --quantize --quant_bits -1


CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR100 --device_ratio S1-W9 --n_split 20 --train_ratio 16-1 --quantize --quant_bits -1