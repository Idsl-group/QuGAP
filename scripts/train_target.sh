#!/bin/bash

# set dataset as per need
# MNIST ("mnist"), Fashion-MNIST ("fmnist") and Transverse-field Ising Model ("ising") supported
dataset="mnist"
tar_names=("conv")

# for 2-class classification -comment if not needed
max=10
step=5
classes=2
qdepths=(10 20)

# for 4-class classification - uncomment if needed
# max=30
# step=10
# classes=4
# qdepths=(20 40 60)

# train classical models - comment this when using Ising dataset
for tar_name in ${tar_names[@]}; do
    python3 ../train_target.py --tar_type="c" --tar_name=$tar_name --dataset=$dataset --num_classes=$classes --img_size=16 --max_epochs=20 --step=10
done

# train quantum models
for qdepth in ${qdepths[@]}; do
    python3 ../train_target.py --tar_type="q" --qdepth=$qdepth --dataset=$dataset --num_classes=$classes --batch_size=64 --max_epochs=$max --step=$step --lr=0.001
done