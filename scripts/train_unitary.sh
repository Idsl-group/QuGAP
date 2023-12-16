#!/bin/bash

tar_class=-1

# MNIST/FMNIST - 2 class 
# qdepths=(10 20)
# lambdas=(1 2 3 4 5 6 7 8 9 10 12 14 15 16 18 20 25 30 35 40 45 50)

# MNIST/FMNIST - 4 class 
# qdepths=(20 40 60)
# lambdas=(1 2 3 4 5 6 7 8 9 10 12 14 15 16 18 20 25 30 35 40 45 50)

# ising - 2 class
qdepths=(10 20)
lambdas=(7.5 8 8.5 9 9.5 10 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15)


# all available settings present - Kindly uncomment and test as you wish
for qdepth in ${qdepths[@]}; do
    for lambd in ${lambdas[@]}; do
        # python3 ../train_unitary.py --qdepth=$qdepth --dataset="fmnist" --num_classes=2 --lambd=$lambd --tar_class=$tar_class --lr=0.001 --max_epochs=15 --step=5 --gamma=0.3
        # python3 ../train_unitary.py --qdepth=$qdepth --dataset="fmnist" --num_classes=4 --lambd=$lambd --tar_class=$tar_class --lr=0.001 --max_epochs=15 --step=5 --gamma=0.3
        python3 ../train_unitary.py --qdepth=$qdepth --dataset="ising" --num_classes=2 --lambd=$lambd --tar_class=$tar_class --lr=0.001 --max_epochs=15 --step=5 --gamma=0.3 --batch_size=25
        # python3 ../train_unitary.py --qdepth=$qdepth --dataset="mnist" --num_classes=2 --lambd=$lambd --tar_class=$tar_class --lr=0.001 --max_epochs=15 --step=5 --gamma=0.3
        # python3 ../train_unitary.py --qdepth=$qdepth --dataset="mnist" --num_classes=4 --lambd=$lambd --tar_class=$tar_class --lr=0.001 --max_epochs=15 --step=5 --gamma=0.3
    done
done