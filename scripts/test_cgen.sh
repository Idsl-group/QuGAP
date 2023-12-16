#!/bin/bash
        
runs=10
gen="full"
tar_class=-1            # we provide only untargeted UAPs (-1), to test targeted UAPs kindly use train_cgen.sh to generate UAPs first
epsilons=(0.30 0.15)    # we provide UAPs trained for these epsilon values, to test other values of epsilon kindly use train_cgen.sh to generate UAPs first
dataset="mnist"         # set dataset (only MNIST/FMNIST supported)

# For MNIST/FMNIST - 2 class (comment as needed)
tar_names=("conv")
qdepths=(10 20)
classes=2

# For MNIST/FMNIST - 4 class (uncomment when needed)
# tar_names=("conv")
# qdepths=(20 40 60)
# classes=4


# To test transferability, change --tes_type and --tes_name/--tes_qdepth to suitable values
# list of trained models is given below:
# Binary classification: ("c" - "conv"), ("q" - 10), ("q" - 20)
# 4class classification: ("c" - "conv"), ("q" - 20), ("q" - 40), ("q" - 60)


for eps in ${epsilons[@]}; do
    # for testing UAPs trained on classical models (comment if not needed)
    for tar_name in ${tar_names[@]}; do
        python3 ../test_cgen.py --gen_name=$gen --runs=$runs --dataset=$dataset --tar_type="c" --tar_name=$tar_name --num_classes=$classes --epsilon=$eps --tar_class=$tar_class --tes_type="c" --tes_name=$tar_name
    done
            
    # for testing UAPs trained on quantum models (comment if not needed)
    for qdepth in ${qdepths[@]}; do
        python3 ../test_cgen.py --gen_name=$gen --runs=$runs --dataset=$dataset --tar_type="q" --qdepth=$qdepth --num_classes=$classes --epsilon=$eps --tar_class=$tar_class --tes_type="q" --tes_qdepth=$qdepth
    done
done