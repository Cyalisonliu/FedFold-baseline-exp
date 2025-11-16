# 2 '#' means needed

# FedAvg, Trainable width ratio = 16:1
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR100 --device_ratio S10 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR100 --device_ratio W10 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR100 --device_ratio S10 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR100 --device_ratio W10 --n_split 40 --train_ratio 16-1 &

# PWM, Trainable width ratio = 16:1 or 8:1 (required: 16:1, n_split = 2, 4, 6)
CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 2 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 2 --train_ratio 16-1 

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 4 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 4 --train_ratio 16-1 & 
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 4 --train_ratio 16-1 

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 6 --train_ratio 16-1 & 
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 6 --train_ratio 16-1 

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 6 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 6 --train_ratio 16-1 & 
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 6 --train_ratio 16-1 

# FWM, Trainable width ratio = 16:1 or 8:1 (fix_split 8.4.2)
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 2 --train_ratio 16-1 --fix_split 2 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 2 --train_ratio 16-1 --fix_split 2 &

# Strong devices only, Trainable width ratio = 16:1
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 2 --train_ratio 16-1 --only_strong &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S10 --n_split 2 --train_ratio 16-1 --only_strong 

# Weak devices only, Trainable width ratio = 16:1
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S9-W1 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S8-W2 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S7-W3 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S6-W4 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S5-W5 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S4-W6 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S3-W7 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S2-W8 --n_split 2 --train_ratio 16-1 --only_weak &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S1-W9 --n_split 2 --train_ratio 16-1 --only_weak &


# FedAvg, Trainable width ratio = 16:8:1
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio W10 --n_split 2 --train_ratio 16-8-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S10 --n_split 2 --train_ratio 16-8-1 &

# PWM, Trainable width ratio = 16:8:1
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S6-M2-W2 --n_split 2 --train_ratio 16-8-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S2-M6-W2 --n_split 2 --train_ratio 16-8-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S2-M2-W6 --n_split 2 --train_ratio 16-8-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S3-M4-W3 --n_split 2 --train_ratio 16-8-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S1-M2-W7 --n_split 2 --train_ratio 16-8-1 &

# PWM, Trainable width ratio = 16:4:1
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S6-M2-W2 --n_split 2 --train_ratio 16-4-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S2-M6-W2 --n_split 2 --train_ratio 16-4-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S2-M2-W6 --n_split 2 --train_ratio 16-4-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S3-M4-W3 --n_split 2 --train_ratio 16-4-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S1-M2-W7 --n_split 2 --train_ratio 16-4-1 &

# FWM, Trainable width ratio = 16:8:1
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S6-M2-W2 --n_split 2 --train_ratio 16-8-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S2-M6-W2 --n_split 2 --train_ratio 16-8-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S2-M2-W6 --n_split 2 --train_ratio 16-8-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S3-M4-W3 --n_split 2 --train_ratio 16-8-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S1-M2-W7 --n_split 2 --train_ratio 16-8-1 --fix_split 8 &

# FWM, Trainable width ratio = 16:8:1
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR10 --device_ratio S6-M2-W2 --n_split 2 --train_ratio 16-4-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S2-M6-W2 --n_split 2 --train_ratio 16-4-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR10 --device_ratio S2-M2-W6 --n_split 2 --train_ratio 16-4-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR10 --device_ratio S3-M4-W3 --n_split 2 --train_ratio 16-4-1 --fix_split 8 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR10 --device_ratio S1-M2-W7 --n_split 2 --train_ratio 16-4-1 --fix_split 8 &


#PWM TinyImageNet
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TinyImageNet --device_ratio S9-W1 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset TinyImageNet --device_ratio S8-W2 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TinyImageNet --device_ratio S7-W3 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TinyImageNet --device_ratio S6-W4 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset TinyImageNet --device_ratio S5-W5 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset TinyImageNet --device_ratio S4-W6 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset TinyImageNet --device_ratio S3-W7 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset TinyImageNet --device_ratio S2-W8 --n_split 40 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset TinyImageNet --device_ratio S1-W9 --n_split 40 --train_ratio 16-1 

# PWM CIFAR100
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR100 --device_ratio S9-W1 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR100 --device_ratio S8-W2 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR100 --device_ratio S7-W3 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR100 --device_ratio S6-W4 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset CIFAR100 --device_ratio S5-W5 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset CIFAR100 --device_ratio S4-W6 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset CIFAR100 --device_ratio S3-W7 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR100 --device_ratio S2-W8 --n_split 20 --train_ratio 16-1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset CIFAR100 --device_ratio S1-W9 --n_split 20 --train_ratio 16-1 &
 
