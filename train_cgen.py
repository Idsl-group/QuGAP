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
    parser.add_argument("--session", default = "Classical Generator Training", type = str, help = "session name")
    parser.add_argument("--tar_type", default = "c", type = str, help = "classical (C) or quantum (Q)")
    parser.add_argument("--tar_name", default = "full", type = str, help = "classical FC-NN (FULL) or CNN (CONV)")
    parser.add_argument("--gen_name", default = "full", type = str, help = "classical FC-NN (FULL) or CNN (CONV)")
    parser.add_argument("--qdepth", default = 20, type=int, help = "depth of the target quantum network")
    
    parser.add_argument("--vic_class", default = -1, type = int, help = "targeted class to be perturbed, -1: general UAP else class-specific UAP")
    parser.add_argument("--tar_class", default = -1, type = int, help = "image perturbed to target class, -1: untargeted UAP else targeted UAP")
    parser.add_argument("--epsilon", default = 0.30, type = float, help = "bound on the L-infty norm of the perturbation generated")

    parser.add_argument("--use_cuda", default = True, type = bool, help = "flag to use cuda when available")
    parser.add_argument("--tar_dir", default = '../targets', type = str, help = "directory to load saved targets")
    parser.add_argument("--save_dir", default = '../uaps', type = str, help = "directory to save generated uaps")
    parser.add_argument("--result_dir", default = "../results", type = str, help = "directory to save all results")
    parser.add_argument("--logs_dir", default = '../logs', type = str, help = "directory to save logs")
    parser.add_argument("--resume", default = False, type = bool, help = "Resuming previous training instance")

    parser.add_argument("--dataset", default = "mnist", type = str, help = "dataset to use: MNIST (mnist), f-MNIST (fmnist) or CIFAR10 (cifar)")
    parser.add_argument("--verbose", default = 1, type = int, help = "verbosity of output")

    parser.add_argument("--lr", default = 0.001, type = float, help = "learning rate for training")
    parser.add_argument("--beta", default = 0.5, type = float, help = "beta1 for Adam optimizer")
    parser.add_argument("--batch_size", default = 64, type = int, help = "batch size for training")
    parser.add_argument("--scheduler", default = "step", type = str, help = "scheduler type - Step LR (step), Reduce LR on Plateau (rlrop)")
    parser.add_argument("--max_epochs", default = 10, type = int, help = "max training epochs")
    parser.add_argument("--step", default = 5, type = int, help = "scheduler step size (for StepLR)")
    parser.add_argument("--gamma", default = 0.1, type = float, help = "LR decay multiplier")
    parser.add_argument("--runs", default=1, type = int, help = "number of repetitions of experiment")
    
    parser.add_argument("--img_size", default = 16, type = int, help = "image size")
    parser.add_argument("--n_channels", default = 1, type = int, help = "number of channels in image")
    parser.add_argument("--num_classes", default = 2, type = int, help = "number of classes of targets")
    parser.add_argument("--class_list", default = None, nargs="+", type = int, help = "list of classes")
    parser.add_argument("--zdim", default = 256, type = int, help = "")

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
    if (args.tar_type).lower() == "c": tar_name = "targ_c_" + (args.tar_name).lower() + "_" + str(len(args.class_list))
    elif (args.tar_type).lower() == "q": tar_name = "targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
    if (args.dataset).lower() == 'fmnist': tar_name = 'fmnist_' + tar_name
    exp_name = exp_name + tar_name + "_" + str(int(args.epsilon * 100)).zfill(2)


    # create results and save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # create and configure logger
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    utils.setup_logging(os.path.join(args.logs_dir, 'train_' + exp_name+'.log'), resume = args.resume)
    logger = logging.getLogger(__name__)
    logger.debug(f'Device: {device}')
    logger.info(f'Session: {args.session}')
    logger.info(f'Classical Generator: {(args.gen_name).upper()}')

    if (args.tar_type).lower() == "c": logger.info(f'Classical Target: {(args.tar_name).upper()}')
    elif (args.tar_type).lower() == "q": logger.info(f'Quantum Target: PQC - Depth {args.qdepth}')

    if vic_class == 'g': logger.info(f'Victim Class: General UAPs')
    else: logger.info(f'Victim Class: {vic_class}')

    if tar_class == 'u': logger.info(f'Target Class: Untargeted Attack')
    else: logger.info(f'Target Class: {tar_class}')

    logger.info(f'Learning Rate: {args.lr}')
    logger.info(f'Epsilon: {args.epsilon}\n')
    logger.debug(f'Arguments: {args}\n')


    # load the trained target model
    logger.debug('Loading target model...')
    if (args.tar_type).lower() == "c":
        if device == torch.device("cpu"): target = torch.load(os.path.join(args.tar_dir, tar_name +'.pt'), map_location = torch.device('cpu'))
        else: target = torch.load(os.path.join(args.tar_dir, tar_name +'.pt'))
    elif (args.tar_type).lower() == "q":
        from qmodels import QNN
        with open(os.path.join(args.tar_dir, tar_name +'.txt'), 'rb') as f:
            target = pickle.load(f)
        if target.U is None:
        # if True:
            target.compute_unitary()
            with open(os.path.join(args.tar_dir, tar_name +'.txt'), 'wb') as f:
                pickle.dump(target, f)
        target = target.to(device)


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
            idxs += utils.get_indices(dataset[i], j)
        sampler = SubsetRandomSampler(idxs)
        dataloader[i] = DataLoader(dataset[i], batch_size = args.batch_size, drop_last = True, sampler = sampler)


    # Fix seed
    seed = 42
    utils.setup_seed(seed)


    # create and configure result directory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, 'train_' + exp_name + '.txt')

    # save stuff
    attacks, gens, losses, noises = [], [], [], []


    # repeat runs
    for run in range(args.runs):
        logger.info(f'Run Number: {run}')

        # load the generator
        if (args.gen_name).lower() == "full": 
            from models import Gen_FC as Gen
        elif (args.gen_name).lower() == "conv": 
            from models import Gen_Conv as Gen
        else: 
            logger.error('Incorrect generator type !')
            raise NotImplementedError("only support classical FC and Conv Generators !")
        gen = Gen(args.zdim, args.img_size, args.n_channels).to(device)

        # training settings
        gen.train()
        optG = optim.Adam(gen.parameters(), lr = args.lr, betas = (args.beta, 0.999))
        sched = optim.lr_scheduler.StepLR(optG, step_size = args.step, gamma = args.gamma)
        criterion = nn.CrossEntropyLoss()
        noise = torch.rand((1,1,args.zdim), device = device)

        counter = 0
        loss = []

        # training loop
        for epoch in range(args.max_epochs):
            t_tot, t_wr = 0, 0
            for i, (data, labels) in enumerate(dataloader['train']):
                
                # generate attack
                real_data = data.to(device)
                attack = args.epsilon * gen(noise).to(device)     # gen output constrained to: (-1,1)
                # print(attack)
                attack = attack.reshape(args.img_size, args.img_size)

                # get the fake datas
                fake_data = real_data + attack
                fake_data = fake_data.reshape(-1, args.img_size * args.img_size)

                # set zerograd
                optG.zero_grad()
                
                # calc probs for classical targets
                if (args.tar_type).lower() == "c":
                    fake_data = torch.clamp(fake_data, min = 0, max = 1)
                    out_logits, out_probs = target(fake_data)
                    out_logits = out_logits.to(device)
                    outs = out_logits

                # calc probs for quantum targets
                elif (args.tar_type).lower() == "q":
                    a_dim = 2**target.a_qubits
                    dim = 2**target.qubits
                    fake_data = torch.div(fake_data, torch.norm(fake_data, dim = 1).reshape(args.batch_size, 1))
                    fake_data = torch.clamp(fake_data, min = 0, max = 1)
                    fake_data = torch.div(fake_data, torch.norm(fake_data, dim = 1).reshape(args.batch_size, 1))
                    fake_data = fake_data.type(torch.cfloat).requires_grad_(True).to(device)

                    # set zero grad and ready for calculation
                    data_fake = fake_data.reshape((-1, 1, dim//a_dim)).clone().requires_grad_(True).to(device)
                    U = target.U.T.clone().requires_grad_(True).to(device)
                    prod = torch.matmul(data_fake, U).mT.clone().requires_grad_(True).to(device)

                    # iterate over the batch
                    out_probs = torch.tensor([], requires_grad = True).to(device)
                    for j, el in enumerate(data_fake):
                        op_probs_c = torch.tensor([], requires_grad = True).to(device)

                        for class_id in range(args.num_classes):
                            op_prob = torch.sum(torch.square(torch.abs(prod[j][torch.arange(class_id, dim, a_dim)]))).clone().requires_grad_(True).to(device)
                            op_probs_c = torch.cat((op_probs_c, op_prob.reshape(1)), 0)

                        op_probs_c = op_probs_c.reshape(1, args.num_classes)/torch.sum(op_probs_c)
                        out_probs = torch.cat((out_probs, op_probs_c), 0)

                    outs = torch.log(out_probs)

                # for untargeted attack
                if args.tar_class == -1:
                    target_label = labels.clone().detach().to(device)
                    loss_adv = - criterion(outs, target_label)

                # for targeted attack
                else:
                    target_label = args.tar_class * torch.ones(labels.shape, dtype = labels.dtype, device = device)
                    loss_adv = criterion(outs, target_label)

                if args.tar_class == -1: 
                    for j, el in enumerate(out_probs):
                        if torch.eq(torch.argmax(el), labels[j]) == 0:
                            t_wr = t_wr + 1
                        t_tot += 1

                loss_adv.backward()
                loss.append(loss_adv.item())            
                optG.step()

                # Show loss values
                counter += 1
                if args.verbose == 2:
                    if counter % 100 == 1:
                        logger.info(f'Iteration: {counter-1}, Fooling Loss: {np.mean(loss[-100:]):0.13f}')
                elif args.verbose == 1:
                    if counter % 100 == 1:
                        logger.debug(f'Iteration: {counter-1}, Fooling Loss: {np.mean(loss[-100:]):0.13f}')
                    if counter % 400 == 1:
                        logger.info(f'Iteration: {counter-1}, Fooling Loss: {np.mean(loss[-400:]):0.13f}')
            sched.step()
        
        # logging
        logger.info(f'Ending training...\n')

        # save generator and attack
        attack = gen(noise)
        attack = torch.min(torch.tensor(1), args.epsilon/torch.norm(attack.flatten(), float('inf')))*attack
        attack = attack.reshape(args.img_size, args.img_size)

        # save stuff
        attacks.append(attack)
        gens.append(gen)
        losses.append(loss)
        noises.append(noise)

    logger.info(f'Saving UAP...\n')
    with open(os.path.join(args.save_dir, exp_name + ".txt"), 'wb') as f:
        pickle.dump((attacks), f)

    with open(os.path.join(args.result_dir, "train_" + exp_name + ".txt"), 'wb') as f:
        pickle.dump((gens, noises, losses), f)