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
    parser.add_argument("--session", default = "Unitary UAP", type = str, help = "session name")
    parser.add_argument("--qdepth", default = 10, type=int, help = "depth of the target quantum network")
    parser.add_argument("--gdepth", default = 50, type=int, help = "depth of the target quantum network")
    parser.add_argument("--clamp_gparams", default = None, type=float, help = "parameter clamping to restrict the local unitaries to be close to identity")
    
    parser.add_argument("--vic_class", default = -1, type = int, help = "targeted class to be perturbed, -1: general UAP else class-specific UAP")
    parser.add_argument("--tar_class", default = -1, type = int, help = "image perturbed to target class, -1: untargeted UAP else targeted UAP")
    parser.add_argument("--lambd", default = 5, type = float, help = "weight given to fidelity loss vs adversarial loss")

    parser.add_argument("--use_cuda", default = True, type = bool, help = "flag to use cuda when available")
    parser.add_argument("--tar_dir", default = '../targets', type = str, help = "directory to load saved targets")
    parser.add_argument("--save_dir", default = '../uaps', type = str, help = "directory to save unitary generators")
    parser.add_argument("--result_dir", default = "../results", type = str, help = "directory to save all results")
    parser.add_argument("--logs_dir", default = '../logs', type = str, help = "directory to save logs")
    parser.add_argument("--resume", default = False, type = bool, help = "Resuming previous training instance")

    parser.add_argument("--dataset", default = "mnist", type = str, help = "dataset to use: MNIST (mnist) or CIFAR10 (cifar)")
    parser.add_argument("--verbose", default = 1, type = int, help = "verbosity of output")

    parser.add_argument("--lr", default = 0.001, type = float, help = "learning rate for training")
    parser.add_argument("--beta", default = 0.5, type = float, help = "beta1 for Adam optimizer")
    parser.add_argument("--batch_size", default = 64, type = int, help = "batch size for training")
    parser.add_argument("--sample", default = 1, type = int, help = "sampling rate for training and testing")
    parser.add_argument("--scheduler", default = "step", type = str, help = "scheduler type - Step LR (step), Reduce LR on Plateau (rlrop)")
    parser.add_argument("--max_epochs", default = 10, type = int, help = "max training epochs")
    parser.add_argument("--step", default = 5, type = int, help = "scheduler step size (for StepLR)")
    parser.add_argument("--gamma", default = 0.1, type = float, help = "LR decay multiplier")
    
    parser.add_argument("--img_size", default = 8, type = int, help = "image size")
    parser.add_argument("--qubits", default = 4, type = int, help = "qubits for Ising model")
    parser.add_argument("--n_channels", default = 1, type = int, help = "number of channels in image")
    parser.add_argument("--num_classes", default = 2, type = int, help = "number of classes of targets")
    parser.add_argument("--class_list", default = None, nargs="+", type = int, help = "list of classes")
    parser.add_argument("--zdim", default = 256, type = int, help = "")

    args = parser.parse_args()

    # device to use
    device = torch.device("cpu")

    # prep class list
    if args.class_list == None: 
        args.class_list = list(range(args.num_classes))


    # set name of experiment
    vic_class = 'g' if args.vic_class == -1 else str(args.vic_class)
    tar_class = 'u' if args.tar_class == -1 else str(args.tar_class)
    if args.clamp_gparams is None: exp_name = vic_class + "_" + tar_class + "_qgen_" + str(args.gdepth) + "_"
    else: exp_name = vic_class + "_" + tar_class + "_qbim_"
    if (args.dataset).lower() == "mnist": 
        tar_name = "targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = int(math.ceil(math.log2(args.n_channels * args.img_size**2)))
        n_qubits += n_a_qubits
        if args.img_size==8: tar_name = str(8) + "_" + tar_name
    elif (args.dataset).lower() == "fmnist": 
        tar_name = "fmnist_targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = int(math.ceil(math.log2(args.n_channels * args.img_size**2)))
        n_qubits += n_a_qubits
        if args.img_size==8: tar_name = str(8) + "_" + tar_name
    elif (args.dataset).lower() == "ising": 
        tar_name = "ising_targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = args.qubits
        n_qubits += n_a_qubits
    else: raise NotImplementedError("only support MNIST, f-MNIST and Ising !")
    if args.clamp_gparams is not None: exp_name = exp_name + tar_name + "_" + str(int(args.clamp_gparams * 100)).zfill(2)
    else: exp_name = exp_name + tar_name + "_" + str(int(args.lambd * 10)).zfill(3) 
    gen_name = exp_name


    # create results and save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # create and configure logger
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    utils.setup_logging(os.path.join(args.logs_dir, 'train_' + exp_name+'.log'), resume = args.resume)
    logger = logging.getLogger(__name__)
    logger.debug(f'Device: {device}')
    if args.clamp_gparams is None: logger.info(f'Session: {args.session}')
    else: logger.info(f'Session: Clamped QBIM with Single Variational Layer')
    logger.info(f'Quantum Generator: PQC - Depth {(args.gdepth)}')
    logger.info(f'Quantum Target: PQC - Depth {args.qdepth}')

    if vic_class == 'g': logger.info(f'Victim Class: General UAPs')
    else: logger.info(f'Victim Class: {vic_class}')

    if tar_class == 'u': logger.info(f'Target Class: Untargeted Attack')
    else: logger.info(f'Target Class: {tar_class}')

    logger.info(f'Learning Rate: {args.lr}')
    if args.clamp_gparams is not None: logger.info(f'Clamp: {args.clamp_gparams}\n')
    else: logger.info(f'Lambda: {args.lambd}\n')
    logger.debug(f'Arguments: {args}\n')


    # load the trained target model
    from qmodels import QNN
    with open(os.path.join(args.tar_dir, tar_name +'.txt'), 'rb') as f:
        target = pickle.load(f)
    if target.U is None:
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
    elif (args.dataset).lower() == "ising":
        from utils import Ising as Dataset
        logger.info(f'Dataset: Ising {args.num_classes}-class classification')
        logger.info(f'Number of Atoms: {args.qubits}\n')
    else: 
        raise NotImplementedError("only support MNIST, f-MNIST and Ising !")


    # prepare the dataloader
    logger.debug('Loading dataset...')
    dataset, dataloader = {}, {}
    if (args.dataset).lower() == "mnist" or (args.dataset).lower() == "fmnist":
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(args.img_size), torchvision.transforms.ToTensor()])
        dataset['train'] = Dataset('../data/', train = True, download = True, transform = transform)
        dataset['test'] = Dataset('../data/', train = False, download = True, transform = transform)
    elif (args.dataset).lower() == "ising":
        dataset['train'] = Dataset('../data/', train = True, num = args.qubits)
        dataset['test'] = Dataset('../data/', train = False, num = args.qubits)

    load_classes = args.class_list if args.vic_class == -1 else [args.vic_class]

    for i in ['train', 'test']:
        idxs = []
        for j in load_classes:
            idx = utils.get_indices(dataset[i], j)
            idxs += idx[:len(idx)//args.sample]
        sampler = SubsetRandomSampler(idxs)
        dataloader[i] = DataLoader(dataset[i], batch_size = args.batch_size, drop_last = True, sampler = sampler)

    # Fix seed
    seed = 42
    utils.setup_seed(seed)


    # create and configure result directory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, 'train_' + exp_name + '.txt')

    # load the generator
    from qmodels import QGen
    qgen = QGen(args.gdepth, args.qdepth, target.q_params.clone().detach(), n_qubits - n_a_qubits, n_qubits, n_a_qubits, args.num_classes, args.clamp_gparams).to(device)

    # training settings
    qgen.train()
    parameters_mod = [{'params':[qgen.gen_params]}]
    optG = optim.Adam(parameters_mod, lr = args.lr, betas = (args.beta, 0.999))
    sched = optim.lr_scheduler.StepLR(optG, step_size = args.step, gamma = args.gamma)
    criterion = nn.CrossEntropyLoss()

    counter = 0
    losses = [[], [], [], []]

    # training loop
    for epoch in range(args.max_epochs):
        for i, (data, labels) in enumerate(dataloader['train']):
            # set zerograd
            optG.zero_grad()

            # generate attack
            real_data = data.to(device).type(torch.complex128)
            fids, preds, _ = qgen(data)
            fidelity = torch.mean(fids)
            outs = torch.log(preds)

            # for untargeted attack
            if args.tar_class == -1:
                target_label = labels.clone().detach().to(device)
                loss_adv = - criterion(outs, target_label)

            # for targeted attack
            else:
                target_label = args.tar_class * torch.ones(labels.shape, dtype = labels.dtype, device = device)
                loss_adv = criterion(outs, target_label)

            # param update
            loss_fid = torch.mean((1-fids)**2)
            if args.clamp_gparams is None:  loss = loss_adv + args.lambd * loss_fid
            else: loss = loss_adv
            loss.backward()          
            optG.step()

            # save values
            losses[0].append(loss_adv.item())
            losses[1].append(loss_fid.item())
            losses[2].append(fidelity.item())
            losses[3].append(loss.item())

            # Show loss values
            counter += 1
            if args.verbose == 2:
                if counter % 20 == 1:
                    logger.info(f'Iteration: {counter-1}, Fooling Loss: {np.mean(losses[0][-20:]):0.10f}, Avg. Fidelity: {100*np.mean(losses[2][-20:]):0.10f}%')
            elif args.verbose == 1:
                if counter % 20 == 1:
                    logger.debug(f'Iteration: {counter-1}, Fooling Loss: {np.mean(losses[0][-20:]):0.10f}, Avg. Fidelity: {100*np.mean(losses[2][-20:]):0.10f}%')
                if counter % 50 == 1:
                    logger.info(f'Iteration: {counter-1}, Fooling Loss: {np.mean(losses[0][-50:]):0.10f}, Avg. Fidelity: {100*np.mean(losses[2][-50:]):0.10f}%')
        
        sched.step()

    # logging
    logger.info(f'Ending training...\n')
    logger.info(f'Saving Quantum UAP...\n')
    with open(os.path.join(args.save_dir, exp_name + ".txt"), 'wb') as f:
        pickle.dump((qgen), f)

    with open(os.path.join(args.result_dir, "train_" + exp_name + ".txt"), 'wb') as f:
        pickle.dump((qgen, losses), f)

    wr, tot, t_wr, t_tot, fid_list = 0, 0, 0, 0, np.array([])
    logger.info(f'Begin testing...')
    with torch.no_grad():
        for i, (data, labels) in (tqdm(enumerate(dataloader['test']), total = len(dataloader['test'])) if args.verbose>=2 else enumerate(dataloader['test'])):
            real_data = data.to(device).type(torch.cfloat)
            fids, fake_preds, true_preds = qgen(real_data, True)

            fid_list = np.append(fid_list, fids.detach().numpy())
            for j, el in enumerate(fake_preds):
                if torch.eq(torch.argmax(true_preds[j]), labels[j]) == 1:
                    if torch.eq(torch.argmax(el), labels[j]) == 0: wr = wr + 1
                    tot += 1
                if torch.eq(torch.argmax(el), labels[j]) == 0: t_wr += 1
                t_tot += 1

    logger.info(f'Wrong Predictions: {wr}')
    logger.info(f'Total Predictions: {tot}')
    miss_rate = (wr/tot)*100
    logger.info(f'Misclassification Rate: {miss_rate}')

    final_acc = (1 - t_wr/t_tot)*100
    logger.info(f'Final Model Accuracy: {final_acc}')
    logger.info(f'Average Fidelity: {100*np.mean(fid_list)}%')
    logger.info(f'Std. Dev. Fidelity: {100*np.std(fid_list)}%\n')