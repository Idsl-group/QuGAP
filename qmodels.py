import torch
import logging
import numpy as np
import logging.config
import torch.nn as nn
import pennylane as qml
from tqdm import tqdm

# enable logging for custom modules
from config import cfg
import utils
logger = logging.getLogger(__name__)


# quantum devices
q_dev_def = qml.device("default.qubit", wires = cfg.N_QUBITS)
q_dev_gen = qml.device("default.qubit", wires = cfg.N_QUBITS - cfg.N_A_QUBITS )
q_dev_ltng = qml.device("lightning.qubit", wires = cfg.N_QUBITS)\


# qnn classifier
@qml.qnode(q_dev_def, interface="torch", diff_method="backprop")
def quantum_classifier(noise, weights, depth, qubits = cfg.N_QUBITS, a_qubits = cfg.N_A_QUBITS, encoded = False):
    weights = weights.reshape(depth, qubits, 3)
    if not encoded:
        qml.AmplitudeEmbedding(features = noise, wires=range(qubits - a_qubits), normalize=True, pad_with = 0)

    for i in range(depth):
        for y in range(qubits):
            qml.Rot(weights[i][y][0], weights[i][y][1], weights[i][y][2], wires = y)
        for y in range(qubits):
            qml.CNOT(wires = [y, (y+1)%qubits])
    return qml.probs(wires=list(range(qubits - a_qubits, qubits)))


# circuit to get unitary
@qml.qnode(q_dev_def, interface="torch", diff_method="backprop")
def quantum_unitary(noise, weights, depth, qubits = cfg.N_QUBITS, a_qubits = cfg.N_A_QUBITS):
    weights = weights.reshape(depth, qubits, 3)
    qml.AmplitudeEmbedding(features = noise, wires=range(qubits), normalize=True, pad_with = 0)
    
    qml.Snapshot("After Encoding")
    for i in range(depth):
        for y in range(qubits):
            qml.Rot(weights[i][y][0], weights[i][y][1], weights[i][y][2], wires = y)
        for y in range(qubits):
            qml.CNOT(wires = [y, (y+1)%qubits])
    qml.Snapshot("After Processing")
    return qml.probs(wires=list(range(qubits - a_qubits, qubits)))


# amplitude encoding circuit
@qml.qnode(q_dev_gen, interface = "torch", diff_method = "backprop")
def amp_encode(inputs, qubits = cfg.N_QUBITS - cfg.N_A_QUBITS):
    qml.AmplitudeEmbedding(features = inputs, wires=range(qubits), normalize=True, pad_with = 0)
    return qml.state()

# fake probs generation
@qml.qnode(q_dev_def, interface = "torch", diff_method = "backprop")
def quantum_fake(noise, g_weights, t_weights, g_depth, t_depth, g_qubits = cfg.N_QUBITS - cfg.N_A_QUBITS, t_qubits = cfg.N_QUBITS, t_a_qubits = cfg.N_A_QUBITS, clamp = False):
    gweights = g_weights.reshape(g_depth, g_qubits, 3)
    tweights = t_weights.reshape(t_depth, t_qubits, 3)
    qml.AmplitudeEmbedding(features = noise, wires=range(g_qubits), normalize=True, pad_with = 0)

    # generating fake samples
    for i in range(g_depth):
        for y in range(g_qubits):
            qml.Rot(gweights[i][y][0], gweights[i][y][1], gweights[i][y][2], wires = y)
        if not clamp:
            for y in range(g_qubits):
                qml.CZ(wires = [y, (y+1)%g_qubits])
    
    # getting fake predictions
    for i in range(t_depth):
        for y in range(t_qubits):
            qml.Rot(tweights[i][y][0], tweights[i][y][1],tweights[i][y][2], wires = y)
        for y in range(t_qubits):
            qml.CNOT(wires = [y, (y+1)%t_qubits])

    return qml.probs(wires=list(range(t_qubits - t_a_qubits, t_qubits)))


# quantum fake state generation
@qml.qnode(q_dev_gen, interface='torch', diff_method="backprop")
def show_fake_img(inputs, weights, depth, qubits = cfg.N_QUBITS - cfg.N_A_QUBITS, clamp = False):
    weights = weights.reshape(depth, qubits, 3)
    qml.AmplitudeEmbedding(features = inputs, wires=range(qubits), normalize=True, pad_with = 0)

    for i in range(depth):
        for y in range(qubits):
            qml.Rot(weights[i][y][0], weights[i][y][1], weights[i][y][2], wires = y)
        if not clamp:
            for y in range(qubits):
                qml.CZ(wires = [y, (y+1)%qubits])
    return qml.state()


# Quantum Neural Network
class QNN(nn.Module):
    """Quantum Neural Network"""
    def __init__(self, depth, device, params = None, qubits = cfg.N_QUBITS, a_qubits = cfg.N_A_QUBITS, im_size = cfg.IMG_SIZE, in_channels = cfg.N_CHANNELS, num_classes = cfg.NUM_CLASSES, mode = 'train'):
        super(QNN, self).__init__()
        self.q_params = None
        self.num_classes = num_classes
        self.im_size = im_size
        self.n_channels = in_channels
        self.qubits = qubits
        self.a_qubits = a_qubits
        self.depth = depth
        self.trained = False
        self.mode = mode
        self.U = None
        self.device = device

        if self.mode == 'train': 
            self.q_params = nn.Parameter((4*np.pi) * torch.rand(depth * qubits * 3), requires_grad=True)
        elif self.mode == 'target' and params is not None:
            self.q_params = nn.Parameter(params, requires_grad = True)
            self.trained = True
        elif self.mode == 'target' and params is None and self.q_params is None:
            logger.error('Warning : Parameters required for target model !!')

        logger.info(f'Target Model: Quantum Variational Circuit with depth {self.depth}')

    def forward(self, x):
        preds = torch.Tensor(0, self.num_classes)
        x = x.reshape((x.shape[0], -1))
        preds = (quantum_classifier(x, self.q_params, self.depth, self.qubits, self.a_qubits, False).float()[:, :self.num_classes])
        preds /= torch.sum(preds, axis = 1).unsqueeze(1)
        return preds
    
    def compute_unitary(self):
        logger.debug('Computing global unitary of the parametrized quantum circuit...')
        dim = 2**(self.qubits)
        uni = np.zeros((dim, dim), dtype = np.cdouble)
        for i in range(dim):
            basis = torch.zeros(1, dim)
            basis[0, i] = 1
            dicter = qml.snapshots(quantum_unitary)(basis.flatten(), self.q_params.clone().detach(), self.depth, self.qubits)
            uni[ : , i] = dicter['After Processing']
        
        self.U = torch.tensor(uni[:, list(range(0, dim, 2**self.a_qubits))], dtype = torch.cfloat)
        return
    
# Quantum Generative Model
class QGen(nn.Module):
    """Quantum Generator + Target Model"""
    def __init__(self, g_depth, t_depth, tar_params, g_qubits = cfg.N_QUBITS - cfg.N_A_QUBITS, t_qubits = cfg.N_QUBITS, t_a_qubits = cfg.N_A_QUBITS, num_classes = cfg.NUM_CLASSES, clamp = None):
        super(QGen, self).__init__()
        self.g_qubits = g_qubits
        self.t_qubits = t_qubits
        self.t_a_qubits = t_a_qubits
        self.g_depth = g_depth
        self.t_depth = t_depth
        self.num_classes = num_classes
        self.clamp = clamp

        self.tar_params = nn.Parameter(tar_params, requires_grad=True)
        self.gen_params = nn.Parameter(0 * torch.rand(g_depth * g_qubits * 3), requires_grad = True)

    def forward(self, x, test=False):
        b_size = x.shape[0]
        x = x.reshape((b_size, -1))
        fids = torch.Tensor(0, 1)
        fake_preds, true_preds = torch.Tensor(0, self.num_classes), None

        enc_x = amp_encode(x)
        if self.clamp is None: fakes = show_fake_img(x, self.gen_params , self.g_depth, self.g_qubits)
        else: fakes = show_fake_img(x, torch.clamp(self.gen_params, min = -self.clamp, max = self.clamp) , self.g_depth, self.g_qubits, clamp=True)
        fids = torch.norm(torch.bmm(enc_x.view(b_size, 1, -1), fakes.view(b_size, -1, 1)), dim=1)
            
        if test: true_preds = quantum_classifier(x, self.tar_params, self.t_depth, self.t_qubits, self.t_a_qubits)
        if self.clamp is None: fake_preds = quantum_fake(x, self.gen_params, self.tar_params, self.g_depth, self.t_depth, self.g_qubits, self.t_qubits, self.t_a_qubits).float()[:,:self.num_classes]
        else: fake_preds = quantum_fake(x, torch.clamp(self.gen_params, min = -self.clamp, max = self.clamp), self.tar_params, self.g_depth, self.t_depth, self.g_qubits, self.t_qubits, self.t_a_qubits, clamp = True).float()[:,:self.num_classes]
        fake_preds /= torch.sum(fake_preds, axis = 1).unsqueeze(1)

        return fids, fake_preds, true_preds