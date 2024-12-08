#!/bin/bash

# Modified by chaoserver: Removed SLURM configs and adapted for CPU training
python train.py --epoch 100 --seed 3 --b 32 --lr 0.0001 --weight_d 0 --gpu 0 \
    --data_path './data/1_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' \
    --save_path 'setting1'

python train.py --epoch 100 --seed 3 --b 32 --lr 0.0001 --weight_d 0 --gpu 0 \
    --data_path './data/2_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' \
    --save_path 'setting2'

python train.py --epoch 100 --seed 3 --b 32 --lr 0.0001 --weight_d 0 --gpu 0 \
    --data_path './data/3_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' \
    --save_path 'setting3'

python train.py --epoch 100 --seed 3 --b 32 --lr 0.0001 --weight_d 0 --gpu 0 \
    --data_path './data/4_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' \
    --save_path 'setting4'

python train.py --epoch 100 --seed 3 --b 32 --lr 0.0001 --weight_d 0 --gpu 0 \
    --data_path './data/5_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' \
    --save_path 'setting5'
