#!/bin/bash

runs=10
tar_class=-1            # set -1 for untargeted UAPs, and class_label to test targeted UAPs (0 or 1 for binary; 0, 1, 2 or 3 for 4-class)
epsilons=(0.1 0.2 0.3)  # we provide UAPs trained for these epsilon values, to test other values of epsilon kindly use train_cgen.sh to generate UAPs first
dataset="mnist"        # only mnist and fmnist are supported

# For MNIST/FMNIST - 2 class (comment if not needed)
tar_names=("conv")
qdepths=(10 20)
classes=2

# For MNIST/FMNIST - 4 class (uncomment if needed)
# tar_names=("conv")
# qdepths=(20 40 60)
# classes=4

# list of trained models is given below:
# Binary classification: ("c" - "conv"), ("q" - 10), ("q" - 20)
# 4class classification: ("c" - "conv"), ("q" - 20), ("q" - 40), ("q" - 60)


# Kindly do not change hyperparameter values for best performance (feel free to experiment though!)
for eps in ${epsilons[@]}; do
    for tar_name in ${tar_names[@]}; do
        python3 ../train_cgen.py --gen_name="full" --dataset=$dataset --runs=$runs --tar_type="c" --tar_name=$tar_name --num_classes=$classes --epsilon=$eps --tar_class=$tar_class --lr=0.005 --max_epochs=15 --step=5 --gamma=0.3
    done

    for qdepth in ${qdepths[@]}; do
        python3 ../train_cgen.py --gen_name="full" --dataset=$dataset --runs=$runs --tar_type="q" --qdepth=$qdepth --num_classes=$classes --epsilon=$eps --tar_class=$tar_class --lr=0.001 --max_epochs=10 --step=4 --gamma=0.5
    done
done