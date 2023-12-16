from easydict import EasyDict

cfg = EasyDict()
cfg.BATCH_SIZE = 256
cfg.N_CHANNELS = 1
cfg.GEN_ZDIM = 256

# KINDLY DO NOT USE ANY SETTINGS EXCEPT THESE

# Setting 1
# ISING: 2 class - only use with train_unitary.sh, train_qbim.sh, train_qgen.sh
# cfg.IMG_SIZE = 4
# cfg.NUM_CLASSES = 2
# cfg.N_A_QUBITS = 1
# cfg.N_QUBITS = 5

# Setting 2
# MNIST/FMNIST 8x8: 2 class - only use with train_unitary.sh, train_qbim.sh, train_qgen.sh
# cfg.IMG_SIZE = 8
# cfg.NUM_CLASSES = 2
# cfg.N_A_QUBITS = 1
# cfg.N_QUBITS = 7

# Setting 3
# MNIST/FMNIST 16x16: 2 class
# cfg.IMG_SIZE = 16
# cfg.NUM_CLASSES = 2
# cfg.N_A_QUBITS = 1
# cfg.N_QUBITS = 9

# Setting 4
# MNIST/FMNIST 16x16: 4 class
cfg.IMG_SIZE = 16
cfg.NUM_CLASSES = 4
cfg.N_A_QUBITS = 2
cfg.N_QUBITS = 10

if __name__ == "__main__":
    print(cfg)