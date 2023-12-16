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
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import cfg
# from qmodels import *
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default = "Target Training", type = str, help = "session name")
    parser.add_argument("--tar_type", default = "c", type = str, help = "classical (C) or quantum (Q)")
    parser.add_argument("--tar_name", default = "full", type = str, help = "classical: FC-NN (FULL) or CNN (CONV); quantum: linear PQC (FULL) or non-linear PQC (CONV)")
    parser.add_argument("--use_cuda", default = True, type = bool, help = "flag to use cuda when available")
    parser.add_argument("--save_dir", default = '../targets', type = str, help = "directory to save trained targets")
    parser.add_argument("--result_dir", default = "../results", type = str, help = "directory to save all results")
    parser.add_argument("--logs_dir", default = '../logs', type = str, help = "directory to save logs")
    parser.add_argument("--resume", default = False, type = bool, help = "Resuming previous training instance")

    parser.add_argument("--dataset", default = "mnist", type = str, help = "dataset to use: MNIST (mnist), f-MNIST (fmnist) or Transverse-field Ising Model (ising)")
    parser.add_argument("--verbose", default = 1, type = int, help = "verbosity of output")

    parser.add_argument("--lr", default = 0.001, type = float, help = "learning rate for training")
    parser.add_argument("--batch_size", default = 256, type = int, help = "batch size for training")
    parser.add_argument("--scheduler", default = "step", type = str, help = "scheduler type - Step LR (step), Reduce LR on Plateau (rlrop)")
    parser.add_argument("--max_epochs", default = 30, type = int, help = "max training epochs")
    parser.add_argument("--step", default = 10, type = int, help = "scheduler step size (for StepLR)")
    parser.add_argument("--gamma", default = 0.3, type = int, help = "LR decay multiplier")

    parser.add_argument("--img_size", default = 16, type = int, help = "image size")
    parser.add_argument("--qubits", default = 4, type = int, help = "qubits for Ising model")
    parser.add_argument("--n_channels", default = 1, type = int, help = "number of channels in image")
    parser.add_argument("--num_classes", default = 2, type = int, help = "number of classes of targets")
    parser.add_argument("--class_list", default = None, nargs="+", type = int, help = "list of classes")
    parser.add_argument("--qdepth", default = 10, type = int, help = "depth of parametrized quantum circuit")

    args = parser.parse_args()


    # device to use
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")


    # prep class list
    if args.class_list == None: 
        args.class_list = list(range(args.num_classes))


    # set name of experiment
    if (args.dataset).lower() =="mnist":
        if (args.tar_type).lower() == "c": tar_name = "targ_c_" + (args.tar_name).lower() + "_" + str(len(args.class_list))
        elif (args.tar_type).lower() == "q": tar_name = "targ_q_" + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = int(math.ceil(math.log2(args.n_channels * args.img_size**2)))
        n_qubits += n_a_qubits
    elif (args.dataset).lower() =="fmnist":
        if (args.tar_type).lower() == "c": tar_name = "fmnist_targ_c_" + (args.tar_name).lower() + "_" + str(len(args.class_list))
        elif (args.tar_type).lower() == "q": tar_name = "fmnist_targ_q_" + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = int(math.ceil(math.log2(args.n_channels * args.img_size**2)))
        n_qubits += n_a_qubits
    elif (args.dataset).lower() =="ising":
        if (args.tar_type).lower() == "c": tar_name = "ising_targ_c_" + (args.tar_name).lower() + "_" + str(len(args.class_list))
        elif (args.tar_type).lower() == "q": tar_name = "ising_targ_q_" + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = args.qubits
        n_qubits += n_a_qubits
    else: 
        raise NotImplementedError("only support MNIST, f-MNIST and Ising !")

    # create results and target directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # create and configure logger
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    utils.setup_logging(os.path.join(args.logs_dir, tar_name+'.log'), resume = args.resume)
    logger = logging.getLogger(__name__)
    logger.debug(f'Device: {device}')
    logger.info(f'Session: {args.session}')
    logger.debug(f'Arguments: {args}\n')


    # load the target model
    if (args.tar_type).lower() == "c":
        if (args.tar_name).lower() == "full": 
            from models import Target_FC as Target
        elif (args.tar_name).lower() == "conv": 
            from models import Target_CNN as Target
        else: 
            logger.error('Incorrect Target Type !')
            raise NotImplementedError("only support classical FC and Conv NNs !")
        target = Target(args.img_size, args.n_channels, args.num_classes).to(device)

    elif (args.tar_type).lower() == "q":
        device = torch.device('cpu')
        from qmodels import QNN as Target
        target = Target(args.qdepth, device, None, n_qubits, n_a_qubits, args.img_size, args.n_channels, args.num_classes).to(device)
        # target.compute_unitary()


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
    logger.info(f'Batch Size: {args.batch_size}')


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

    for i in ['train', 'test']:
        idxs = []
        for j in args.class_list:
            idxs += utils.get_indices(dataset[i], j)
        sampler = SubsetRandomSampler(idxs)
        dataloader[i] = DataLoader(dataset[i], batch_size = args.batch_size, drop_last = True, sampler = sampler)

    # Fix seed
    seed = 42
    utils.setup_seed(seed)
    

    # begin training
    opt = torch.optim.Adam(target.parameters(), lr = args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, args.step, args.gamma)
    criterion = nn.CrossEntropyLoss()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    epoch = -1

    # create and configure result directory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, tar_name + '.txt')

    if not args.resume:
        with open(result_path, 'wb') as f:
            pickle.dump((target, opt, train_loss, train_acc, val_loss, val_acc, epoch),f)

    # training loop
    logger.debug('Training the model...')
    for epoch in range(args.max_epochs):

        # load data before epoch
        if args.resume:
            with open(result_path, 'rb') as f:
                (target, opt, train_loss, train_acc, val_loss, val_acc, epoch) = pickle.load(f)
            epoch += 1
        if args.verbose >= 1:
            logger.info(f"Epoch {epoch+1} of {args.max_epochs}: Learning Rate {sched.get_last_lr()[0]:.5f}")

        # training set
        loss, acc, tot = 0.0, 0, 0
        for i, (data, labels) in tqdm(enumerate(dataloader['train']), total = len(dataloader['train'])) if args.verbose >= 2 else enumerate(dataloader['train']):
            opt.zero_grad()
            data, labels = data.to(device), labels.to(device)
            if (args.tar_type).lower() == 'c': 
                logits, probs = target(data)
            elif (args.tar_type).lower() == 'q': 
                probs = target(data)
                logits = torch.log(probs)
            
            loss_batch = criterion(logits, labels)
            loss_batch.backward()
            opt.step()


            loss += loss_batch.data.item()
            x, y = utils.accuracy(probs, labels)
            acc, tot = acc + x, tot + y

        sched.step()
        train_loss.append(loss * args.batch_size / tot)
        train_acc.append(acc * 100 / tot)
        if args.verbose >= 1:
            logger.info(f"Training Loss: {train_loss[-1]:.8f}, Accuracy: {train_acc[-1]:.5f} %")

        # validation set
        loss, acc, tot = 0.0, 0, 0
        for i, (data, labels) in tqdm(enumerate(dataloader['test']), total = len(dataloader['test'])) if args.verbose >= 2 else enumerate(dataloader['test']):
            opt.zero_grad()
            data, labels = data.to(device), labels.to(device)
            with torch.no_grad():
                if (args.tar_type).lower() == 'c': 
                    logits, probs = target(data)
                elif (args.tar_type).lower() == 'q': 
                    probs = target(data)
                    logits = torch.log(probs)
                
                loss_batch = criterion(logits, labels)
                loss += loss_batch.data.item()
                x, y = utils.accuracy(probs, labels)
                acc, tot = acc + x, tot + y

        val_loss.append(loss * args.batch_size  / tot)
        val_acc.append(acc * 100 / tot)
        if args.verbose >= 1:
            logger.info(f"Testing Loss: {val_loss[-1]:.8f}, Accuracy: {val_acc[-1]:.5f} %\n")
        
        # save stuff
        with open(result_path, 'wb') as f:
            pickle.dump((target, opt, train_loss, train_acc, val_loss, val_acc, epoch),f)
    logger.debug('Ending training...')
    

    logger.debug('Saving target model...')
    if (args.tar_type).lower() == "c": torch.save(target, os.path.join(args.save_dir, tar_name + ".pt"))
    elif (args.tar_type).lower() == "q": 
        with open(os.path.join(args.save_dir, tar_name + ".txt"), 'wb') as f:
            pickle.dump(target, f)
