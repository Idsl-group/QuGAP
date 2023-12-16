#!/bin/bash

qdepths=(10 20)                 # setting depths for quantum classifiers (use only these settings - unless you train targets yourself)
clamps=(0.1 0.2 0.3 0.4 0.5)    # parameter clamping (feel free to adjust stuff)
tar_class=-1                    # redundant - targeted attacks not well defined for 2 class classification

# Ising and MNIST-8x8 supported - uncomment for use
for clamp in ${clamps[@]}; do
    for qdepth in ${qdepths[@]}; do
        python3 ../train_qgen.py --dataset="ising" --num_classes=2 --qdepth=$qdepth --gdepth=1 --clamp_gparams=$clamp --lr=0.01 --max_epochs=10 --step=4 --gamma=1 --batch_size=500
        #python3 ../train_qgen.py --dataset="mnist" --num_classes=2 --qdepth=$qdepth --gdepth=1 --clamp_gparams=$clamp --lr=0.01 --max_epochs=10 --step=4 --gamma=1 --batch_size=64
    done
done