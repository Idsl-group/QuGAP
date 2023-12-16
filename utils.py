######################################################
#  UTILITY FUNCTIONS TO HELP WITH STUFF
######################################################
# SETTING UP LOGGING FOR FILE.log & STDOUT
# SETTING SEED FOR REPRODUCIBILITY
# DENSITY MATRIX & FIDELITY CALCULATION (Simulation)
# Dataset class for ISING DATASET
######################################################


# Library imports
import sys
import time
import torch
import random
import logging
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)

def density_matrix(state):
    state = state.unsqueeze(0)
    out = torch.matmul(np.conj(state).T, state)
    return out

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item(), float(len(labels))


#redundant
def fid_loss(fid_list, eta = 0.80, conf_bound = 0.01, scaler = 10**12, exp = 10):
    fid_loss = 0.0 
    for gen_fid in fid_list:
        fid_loss = fid_loss + (((gen_fid - eta)/conf_bound)**exp)/scaler
    fid_loss /= len(fid_list)
    return fid_loss

# simulate classical fidelity calculation btwn images
def fid_imgs(img, fake):
    b_size = img.shape[0]
    img, fake = img.reshape(b_size, -1), fake.reshape(b_size, -1)
    norm_img = torch.divide(img, torch.norm(img, dim = 1).reshape(b_size, -1))
    norm_fake = torch.divide(fake, torch.norm(fake, dim = 1).reshape(b_size, -1))
    fids = torch.abs(torch.bmm(norm_img.view(b_size, 1, -1), norm_fake.view(b_size, -1, 1))**2)
    return fids


# set seed for all calculations
def setup_seed(seed, level = 1):
    logger.debug('Setting seed for reproducibility...')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if level >= 1:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# setup logger
def setup_logging(filename = None, resume = False):
    root_logger = logging.getLogger()

    console = logging.StreamHandler(sys.stdout)
    file = logging.FileHandler(filename = filename, mode = 'a' if resume else 'w')

    root_logger.setLevel(logging.DEBUG)
    console.setLevel(logging.INFO)
    file.setLevel(logging.DEBUG)

    chformatter = logging.Formatter("%(asctime)s ==> %(message)s", "%m/%d/%Y %I:%M:%S %p")
    fhformatter = logging.Formatter("%(asctime)s : %(name)-12s %(levelname)-8s ==> %(message)s", "%m/%d/%Y %I:%M:%S %p")
    console.setFormatter(chformatter)
    file.setFormatter(fhformatter)

    root_logger.addHandler(console)
    root_logger.addHandler(file)


# dataset class for Ising model
class Ising(Dataset):
    def __init__(self, data_path = '../data/', train = True, num = 4, transform = None):
        self.transform = transform
        phase = 'train' if train else 'test'
        self.pickle_path = data_path + 'ising/'+ str(num) + '/transverse_field_ising_model_' + phase + '.csv'
        self.ising_frame = pd.read_pickle(self.pickle_path)
        self.targets = self.ising_frame['Phase']
        
    def __len__(self):
        return len(self.ising_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        state = torch.tensor(self.ising_frame['Ground State'][idx], dtype=torch.cfloat)
        phase = torch.tensor(self.ising_frame['Phase'][idx], dtype=torch.long)
        sample = (state, phase)
        if self.transform:
            sample = self.transform(sample)
        return sample