#!/bin/bash

# test out binary classification on Ising and MNIST
qdepths=(10 20)
tar_class=-1

# Increasing gdepth increases execution time!
# gdepth - set such that atleast d^2 parameters in the circuit (where d is dimensionality of data)

# for ising - (feel free to test out other values and explore!)
lambdas=(20 22 24 26 28 30)
gdepths=(30)

# for mnist - (use depths >100 for best performance - very slow training)
lambdas=(20 22 24 26 28 30)
gdepths=(200)

# Ising and MNIST-8x8 supported - uncomment for use
for gdepth in ${gdepths[@]}; do
    for qdepth in ${qdepths[@]}; do
        for lambda in ${lambdas[@]}; do
            python3 ../train_qgen.py --dataset="ising" --num_classes=2 --qdepth=$qdepth --gdepth=$gdepth --lambd=$lambda --tar_class=$tar_class --lr=0.001 --max_epochs=10 --step=4 --gamma=0.3 --batch_size=50
            # python3 ../train_qgen.py --dataset="mnist" --num_classes=2 --qdepth=$qdepth --gdepth=$gdepth --lambd=$lambda --tar_class=$tar_class --lr=0.001 --max_epochs=10 --step=4 --gamma=0.5 --batch_size=64 --sample=10
        done
    done
done
