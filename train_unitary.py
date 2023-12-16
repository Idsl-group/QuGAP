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
    parser.add_argument("--session", default = "Unitary Training Simulation", type = str, help = "session name")
    parser.add_argument("--qdepth", default = 20, type=int, help = "depth of the target quantum network")
    
    parser.add_argument("--vic_class", default = -1, type = int, help = "targeted class to be perturbed, -1: general UAP else class-specific UAP")
    parser.add_argument("--tar_class", default = -1, type = int, help = "image perturbed to target class, -1: untargeted UAP else targeted UAP")
    parser.add_argument("--lambd", default = 1, type =float, help = "weight given to fidelity loss term")

    parser.add_argument("--use_cuda", default = True, type = bool, help = "flag to use cuda when available")
    parser.add_argument("--tar_dir", default = '../targets', type = str, help = "directory to load saved targets")
    parser.add_argument("--save_dir", default = '../uaps', type = str, help = "directory to save trained unitary")
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
    
    parser.add_argument("--img_size", default = 16, type = int, help = "image size")
    parser.add_argument("--qubits", default = 4, type = int, help = "qubits for Ising model")
    parser.add_argument("--n_channels", default = 1, type = int, help = "number of channels in image")
    parser.add_argument("--num_classes", default = 2, type = int, help = "number of classes of target")
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
    exp_name = vic_class + "_" + tar_class + "_quni_"
    if (args.dataset).lower() =="mnist": 
        tar_name = "targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = int(math.ceil(math.log2(args.n_channels * args.img_size**2)))
        n_qubits += n_a_qubits
        n_dim  = args.img_size**2
        if args.img_size==8: tar_name = str(8) + "_" + tar_name
    elif (args.dataset).lower() =="fmnist": 
        tar_name = "fmnist_targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = int(math.ceil(math.log2(args.n_channels * args.img_size**2)))
        n_qubits += n_a_qubits
        n_dim  = args.img_size**2
        if args.img_size==8: tar_name = str(8) + "_" + tar_name
    elif (args.dataset).lower() =="ising": 
        tar_name = "ising_targ_q_"  + str(args.qdepth) + "_" + str(len(args.class_list))
        n_a_qubits = int(math.ceil(math.log2(args.num_classes)))
        n_qubits = args.qubits
        n_qubits += n_a_qubits
        n_dim = 2**(args.qubits)
    else: raise NotImplementedError("only support MNIST, f-MNIST and Ising !")
    exp_name = exp_name + tar_name + "_" + str(int(args.lambd*10)).zfill(3)


    # create results and save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # create and configure logger
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    utils.setup_logging(os.path.join(args.logs_dir, 'train_' + exp_name + '.log'), resume = args.resume)
    logger = logging.getLogger(__name__)
    logger.debug(f'Device: {device}')
    logger.info(f'Session: {args.session}')
    logger.info(f'Quantum Target: PQC - Depth {args.qdepth}')

    if vic_class == 'g': logger.info(f'Victim Class: General UAPs')
    else: logger.info(f'Victim Class: {vic_class}')
    if tar_class == 'u': logger.info(f'Target Class: Untargeted Attack')
    else: logger.info(f'Target Class: {tar_class}')
    
    logger.info(f'Learning Rate: {args.lr}')
    logger.info(f'Lambda: {args.lambd}\n')
    logger.debug(f'Arguments: {args}\n')

    # load the trained target model
    logger.debug('Loading target model...')
    from qmodels import QNN
    with open(os.path.join(args.tar_dir, tar_name +'.txt'), 'rb') as f:
        target = pickle.load(f)
    if target.U is None:
    # if True:
        target.compute_unitary()
        with open(os.path.join(args.tar_dir, tar_name +'.txt'), 'wb') as f:
            pickle.dump(target, f)
    target = target.to(device)

    # set dims
    a_dim = 2**target.a_qubits
    dim = 2**target.qubits

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
    if (args.dataset).lower() == "mnist" or (args.dataset).lower() == "fmnist" :
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


    # load the generator
    from models import Gen_Uni as Gen
    gen = Gen(n_dim).to(device)

    # training settings
    gen.train()
    optG = optim.Adam(gen.parameters(), lr = args.lr, betas = (args.beta, 0.999))
    sched = optim.lr_scheduler.StepLR(optG, step_size = args.step, gamma = args.gamma)
    criterion = nn.CrossEntropyLoss()

    counter = 0
    loss = [[], [], []]

    # training loop
    logger.info('Begin training...')
    for epoch in range(args.max_epochs):
        acc, tot = 0., 0.
        for i, (data, labels) in enumerate(dataloader['train']):
            
            # generate attack
            real_data = data.to(device).type(torch.cfloat).reshape(-1, n_dim)
            fake_data = gen(real_data).to(device).type(torch.cfloat).reshape(-1, n_dim)

            # set zerograd
            optG.zero_grad()

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


            # calculate adversarial loss for untargeted attack
            if args.tar_class == -1:
                target_label = labels.clone().detach().to(device)
                loss_adv = - criterion(outs, target_label)

            # calculate adversarial loss for targeted attack
            else:
                target_label = args.tar_class * torch.ones(labels.shape, dtype = labels.dtype, device = device)
                loss_adv = criterion(outs, target_label)

            # calculate fidelity loss
            fids = utils.fid_imgs(real_data, fake_data)
            loss_fid = torch.mean((1 - fids)**2)

            # calculate total loss 
            loss_tot = loss_adv + args.lambd * loss_fid
            loss_tot.backward()
            optG.step()

            # Show loss values
            counter += 1
            loss[0].append(loss_tot.item())
            loss[1].append(loss_adv.item())
            loss[2].append(torch.mean(fids).item())

            if args.verbose == 2:
                if counter % 100 == 1:
                    logger.info(f'Iteration: {counter-1}, Loss: {np.mean(loss[0][-100:]):0.6f}, Fooling Loss: {np.mean(loss[1][-100:]):0.6f}, Avg. Fidelity: {100*np.mean(loss[2][-100:]):0.6f}%')
            elif args.verbose == 1:
                if counter % 100 == 1:
                    logger.debug(f'Iteration: {counter-1}, Loss: {np.mean(loss[0][-100:]):0.6f}, Fooling Loss: {np.mean(loss[1][-100:]):0.6f}, Avg. Fidelity: {100*np.mean(loss[2][-100:]):0.6f}%')
                if counter % 400 == 1:
                    logger.info(f'Iteration: {counter-1}, Loss: {np.mean(loss[0][-400:]):0.6f}, Fooling Loss: {np.mean(loss[1][-400:]):0.6f}, Avg. Fidelity: {100*np.mean(loss[2][-400:]):0.6f}%')
        sched.step()

    # logging
    logger.info(f'Ending training...')

    # save unitary
    mat = gen.uni.clone().cpu().detach()
    u, _, v = torch.svd(mat)
    uni = u @ (v.H)

    logger.info(f'Saving Unitaries...\n')
    with open(os.path.join(args.save_dir, exp_name + ".txt"), 'wb') as f:
        pickle.dump((uni), f)

    with open(os.path.join(args.result_dir, "train_" + exp_name + ".txt"), 'wb') as f:
        pickle.dump((gen, loss), f)

    # begin testing
    logger.info('Begin testing...')
    tot, wr, tar, fid, std = 0, 0, 0, [], []
    t_tot, t_wr = 0, 0

    # testing loop
    device = torch.device("cpu")
    for i, (data, label) in enumerate(dataloader['test']):
        
        # set nograd
        with torch.no_grad():

            # generate attack
            gen = gen.to(device)
            real_data = data.to(device).type(torch.cfloat).reshape(-1, n_dim)
            fake_data = gen(real_data).to(device).type(torch.cfloat).reshape(-1, n_dim)
            target = target.to(device)

            # get prediction probabilities
            real_preds = target(real_data)
            fake_preds = target(fake_data)
            fid.append(torch.mean(utils.fid_imgs(real_data, fake_data)).item())
            std.append(torch.std(utils.fid_imgs(real_data, fake_data)).item())

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

    logger.info(f'Wrong Predictions: {wr}')
    if args.tar_class != -1:
        logger.info(f'Targeted Mispredictions: {tar}')
    logger.info(f'Total Predictions: {tot}\n')
    
    if args.tar_class == -1: 
        miss_rate = (wr/tot)*100
        logger.info(f'Misclassification Rate: {miss_rate}')
    else:
        miss_rate = (tar/tot)*100 
        logger.info(f'Targeted Misclassification Rate: {miss_rate}')

    avg_fid = (np.mean(fid))*100
    avg_std = (np.mean(std))*100
    final_acc = (1 - t_wr/t_tot)*100
    logger.info(f'Final Model Accuracy: {final_acc}')
    logger.info(f'Average Fidelity: {avg_fid}')
    logger.info(f'Std. Dev Fidelity: {avg_std}\n\n')