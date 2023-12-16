# import stuff
import os
import time
import math
import random
import pickle
import logging
import logging.config
import argparse

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import cfg
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default = "UAP Testing: Classical Generator", type = str, help = "session name")
    parser.add_argument("--tar_type", default = "c", type = str, help = "classical (C) or quantum (Q)")
    parser.add_argument("--tar_name", default = "full", type = str, help = "classical FC-NN (FULL) or CNN (CONV)")
    parser.add_argument("--qdepth", default = 20, type=int, help = "depth of the target quantum network")
    parser.add_argument("--tes_type", default = "c", type = str, help = "classical (C) or quantum (Q) or noisy quantum (N)")
    parser.add_argument("--tes_name", default = "full", type = str, help = "classical FC-NN (FULL) or CNN (CONV)")
    parser.add_argument("--tes_qdepth", default = 20, type=int, help = "depth of the test quantum network")
    parser.add_argument("--gen_name", default = "full", type = str, help = "classical FC-NN (FULL) or CNN (CONV)")
    parser.add_argument("--best", default=0, type = int, help = "Best performing attack")
    
    parser.add_argument("--vic_class", default = -1, type = int, help = "targeted class to be perturbed, -1: general UAP else class-specific UAP")
    parser.add_argument("--tar_class", default = -1, type = int, help = "image perturbed to target class, -1: untargeted UAP else targeted UAP")
    parser.add_argument("--epsilon", default = 0.30, type = float, help = "bound on the L-infty norm of the perturbation generated")

    parser.add_argument("--use_cuda", default = True, type = bool, help = "flag to use cuda when available")
    parser.add_argument("--tar_dir", default = '../targets', type = str, help = "directory to load saved targets")
    parser.add_argument("--save_dir", default = '../uaps', type = str, help = "directory to load saved UAPs")
    parser.add_argument("--result_dir", default = "../results", type = str, help = "directory to save all results")
    parser.add_argument("--logs_dir", default = '../logs', type = str, help = "directory to save logs")
    parser.add_argument("--resume", default = False, type = bool, help = "Resuming previous training instance")
    parser.add_argument("--runs", default=1, type = int, help = "number of repetitions of experiment")

    parser.add_argument("--dataset", default = "mnist", type = str, help = "dataset to use: MNIST (mnist), f-MNIST (fmnist) or CIFAR10 (cifar)")
    parser.add_argument("--verbose", default = 1, type = int, help = "verbosity of output")

    parser.add_argument("--batch_size", default = 64, type = int, help = "batch size for testing")
    parser.add_argument("--img_size", default = 16, type = int, help = "image size")
    parser.add_argument("--n_channels", default = 1, type = int, help = "number of channels in image")
    parser.add_argument("--num_classes", default = 4, type = int, help = "number of classes of targets")
    parser.add_argument("--class_list", default = None, nargs="+", type = int, help = "list of classes")

    args = parser.parse_args()

    # device to use
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")


    # prep class list
    if args.class_list == None: 
        args.class_list = list(range(args.num_classes))


    # set name of experiment
    vic_class = 'g' if args.vic_class == -1 else str(args.vic_class)
    tar_class = 'u' if args.tar_class == -1 else str(args.tar_class)

    exp_name = vic_class + "_" + tar_class + "_cgen_" + (args.gen_name).lower() + "_"
    if (args.tes_type).lower() == "c": tes_name = "targ_c_" + (args.tes_name).lower() + "_" + str(len(args.class_list))
    elif (args.tes_type).lower() == "q": tes_name = "targ_q_"  + str(args.tes_qdepth) + "_" + str(len(args.class_list))
    if (args.dataset).lower() == 'fmnist': tes_name = 'fmnist_' + tes_name
    

    if (args.tar_type).lower() == "c": tar_name = "targ_c_" + (args.tar_name).lower() + "_" + str(len(args.class_list))
    elif (args.tar_type).lower() == "q": tar_name = "targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
    if (args.dataset).lower() == 'fmnist': tar_name = 'fmnist_' + tar_name

    gen_name = exp_name + tar_name + "_" + str(int(args.epsilon * 100)).zfill(2)
    exp_name = exp_name + tar_name + "_" + tes_name + "_" + str(int(args.epsilon * 100)).zfill(2)


    # create results and save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # create and configure logger
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    utils.setup_logging(os.path.join(args.logs_dir,'test_' + exp_name +'.log'), resume = args.resume)
    logger = logging.getLogger(__name__)
    logger.info(f'Device: {device}')
    logger.info(f'Session: {args.session}')
    logger.info(f'Classical Generator: {(args.gen_name).upper()}')

    if (args.tar_type).lower() == "c": logger.info(f'Trained on Classical: {(args.tar_name).upper()}')
    elif (args.tar_type).lower() == "q": logger.info(f'Trained on Quantum: PQC {args.qdepth}')
    if (args.tes_type).lower() == "c": logger.info(f'Classical Target: {(args.tes_name).upper()}')
    elif (args.tes_type).lower() == "q": logger.info(f'Quantum Target: PQC {args.tes_qdepth}')

    if vic_class == 'g': logger.info(f'Victim Class: General UAPs')
    else: logger.info(f'Victim Class: {vic_class}')

    if tar_class == 'u': logger.info(f'Target Class: Untargeted Attack')
    else: logger.info(f'Target Class: {tar_class}')

    logger.info(f'Epsilon: {args.epsilon}')
    logger.debug(f'Arguments: {args}\n')


    # load the trained target model
    logger.debug('Loading target model...')
    if (args.tes_type).lower() == "c":
        if device == torch.device("cpu"): target = torch.load(os.path.join(args.tar_dir, tes_name +'.pt'), map_location = torch.device('cpu'))
        else: target = torch.load(os.path.join(args.tar_dir, tes_name +'.pt'), map_location = device)
    elif (args.tes_type).lower() == "q":
        device = torch.device('cpu')
        from qmodels import QNN
        with open(os.path.join(args.tar_dir, tes_name +'.txt'), 'rb') as f:
            target = pickle.load(f).to(device)


    # load dataset
    if (args.dataset).lower() == "mnist":
        from torchvision.datasets import MNIST as Dataset
        logger.info(f'Dataset: MNIST {args.num_classes}-class classification')
        logger.info(f'Image Size: {args.img_size}x{args.img_size}\n')
    elif (args.dataset).lower() == "fmnist":
        from torchvision.datasets import FashionMNIST as Dataset
        logger.info(f'Dataset: Fashion-MNIST {args.num_classes}-class classification')
        logger.info(f'Image Size: {args.img_size}x{args.img_size}\n')
    else: 
        raise NotImplementedError("only support MNIST and f-MNIST!")


    # prepare the training dataloader
    logger.debug('Loading dataset...')
    dataset, dataloader = {}, {}
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(args.img_size), torchvision.transforms.ToTensor()])
    dataset['train'] = Dataset('../data/', train = True, download = True, transform = transform)
    dataset['test'] = Dataset('../data/', train = False, download = True, transform = transform)
    load_classes = args.class_list if args.vic_class == -1 else [args.vic_class]
    for i in ['train', 'test']:
        idxs = []
        for j in load_classes:
            idx = utils.get_indices(dataset[i], j)
            idxs += idx[:len(idx)]
        sampler = SubsetRandomSampler(idxs)
        dataloader[i] = DataLoader(dataset[i], batch_size = args.batch_size, drop_last = True, sampler = sampler)


    # Fix seed
    seed = 42
    utils.setup_seed(seed)

    # create and configure result directory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, 'test_' + exp_name + '.txt')
    runs = list(range(args.runs))
    attacks, miss_rates, final_accs, fids = [ None ] * args.runs, np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    # load the generated UAP
    logger.debug('Loading generated UAP...')
    if args.epsilon != 0:
        with open(os.path.join(args.save_dir, gen_name + '.txt'), 'rb') as f:
            attacks = pickle.load(f)

    best_miss_rate = 0
    best_run = 0

    for run in runs:
        logger.info(f'Run Number: {run}' if run+1 != args.runs else f'Run Number: {run}\n')    
        with torch.no_grad():
            # set target and attack
            if args.epsilon == 0.0: attack = torch.zeros(args.img_size, args.img_size)
            else: attack = attacks[run].to(device)
            target = target.to(device)
            tot, wr, tar, fid = 0, 0, 0, 0
            t_tot, t_wr = 0,0

            # testing loop
            for i, (data, label) in tqdm(enumerate(dataloader['test']), total = len(dataloader['test'])) if args.verbose >= 2 else enumerate(dataloader['test']):
                # generate attack
                real_data = data.to(device)
                label = label.to(device)
            
                # get the fake data
                fake_data = real_data + attack
                fake_data = fake_data.reshape(-1, args.img_size * args.img_size)
                fid += torch.sum(utils.fid_imgs(real_data, fake_data))

                # get prediction probabilities
                if (args.tes_type).lower() == 'c': 
                    fake_data = torch.clamp(fake_data, min = 0, max = 1)
                    _, real_preds = target(real_data)
                    _, fake_preds = target(fake_data)
                else: 
                    real_preds = target(real_data)
                    fake_data = torch.div(fake_data, torch.norm(fake_data, dim = 1).reshape(args.batch_size, 1))
                    fake_data = torch.clamp(fake_data, min = 0, max = 1)
                    fake_preds = target(fake_data)

                # untargeted attack
                if args.tar_class == -1: 
                    for j, el in enumerate(fake_preds):
                        if torch.eq(torch.argmax(real_preds[j]), label[j]) == 1:
                            if torch.eq(torch.argmax(el), label[j]) == 0: wr, tar = wr + 1, tar + 1
                            tot += 1
                        if torch.eq(torch.argmax(el), label[j]) == 0: t_wr = t_wr + 1
                        t_tot += 1

                # targeted attacks
                else: 
                    target_label = args.tar_class * torch.ones(label.shape, dtype = label.dtype, device = device)
                    for j, el in enumerate(fake_preds):
                        if torch.eq(torch.argmax(real_preds[j]), label[j]) == 1:
                            if torch.eq(torch.argmax(el), label[j]) == 0: wr += 1
                            if torch.eq(torch.argmax(el), target_label[j]) == 1: tar += 1
                            tot += 1

                        if torch.eq(torch.argmax(el), label[j]) == 0: t_wr += 1
                        t_tot += 1

        logger.debug(f'Wrong Predictions: {wr}')
        if args.tar_class != -1:
            logger.debug(f'Targeted Mispredictions: {tar}')
        logger.debug(f'Total Predictions: {tot}')
        
        if args.tar_class == -1: 
            miss_rate = (wr/tot)*100
            logger.debug(f'Misclassification Rate: {miss_rate}')
        else:
            miss_rate = (tar/tot)*100 
            logger.debug(f'Targeted Misclassification Rate: {miss_rate}')

        avg_fid = (fid/t_tot)*100
        final_acc = (1 - t_wr/t_tot)*100
        logger.debug(f'Final Model Accuracy: {final_acc}')
        #logger.debug(f'Average Fidelity: {avg_fid}\n')

        # collect info
        attacks[run] = attack
        miss_rates[run] = miss_rate
        final_accs[run] = final_acc
        fids[run] = avg_fid

        if miss_rate > best_miss_rate:
            best_run = run
            best_miss_rate = miss_rate

    # print best results
    logger.info('Best Attack Stats')
    logger.info(f'Best Run: {best_run}')
    if args.tar_class == -1: logger.info(f'Misclassification Rate: {miss_rates[best_run]}')
    else: logger.info(f'Targeted Misclassification Rate: {miss_rates[best_run]}')
    #logger.info(f'Average Fidelity: {fids[best_run]}')
    logger.info(f'Final Model Accuracy: {final_accs[best_run]}\n')

    # print avg statistics
    logger.info('Average Attack Stats')
    if args.tar_class == -1: logger.info(f'Misclassification Rate')
    else: logger.info(f'Targeted Misclassification Rate')
    logger.info(f'Mean: {np.mean(miss_rates)}')
    logger.info(f'Std Dev: {np.std(miss_rates)}')
    logger.info(f'Final Model Accuracy')
    logger.info(f'Mean: {np.mean(final_accs)}')
    logger.info(f'Std Dev: {np.std(final_accs)}\n\n')

    # save results
    with open(os.path.join(args.result_dir, "test_" + exp_name + "_all.txt"), 'wb') as f:
        pickle.dump((attacks, miss_rates, final_accs, fids), f)

    # save results
    if (args.tes_type).lower() != "n":
        with open(os.path.join(args.result_dir, "test_" + exp_name + "_best.txt"), 'wb') as f:
            pickle.dump((attacks[best_run], miss_rates[best_run], final_accs[best_run], fids[best_run]), f)
        with open(os.path.join(args.result_dir, "test_" + exp_name + "_avgs.txt"), 'wb') as f:
            pickle.dump((np.mean(miss_rates), np.std(miss_rates), np.mean(final_accs), np.std(final_accs), np.mean(fids), np.std(fids)), f)